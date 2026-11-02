"""Generate predictions on pinned validation data and average training seeds.

Provide R² quality filtering, held-out/OOF RAE-ratio filtering, and
correlation-based model deduplication for ensemble selection.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from data_tools.inputs import INPUT_REGISTRY, featurize, is_known_input
from models import REGISTRY
from models.oof_chemprop import (
    _resolve_chemprop_config,
    _split_internal_validation,
)
from models.training.chemprop import build_command
from models.utils import drop_nan_rows


_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

def validation_prediction(
    df: pl.DataFrame,
    model_name: str, 
    model_id: str, 
    model_params: dict,
    input_names: list[str],
    targets: list[str],
    cache_dir: Path, 
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """This handles feature-based models only and returns only validation
    rows with observed finite labels.
    """

    if "split" not in df.columns:
        raise ValueError("split column must exist")

    train_df = df.filter(pl.col("split") == "train")
    val_df = df.filter(pl.col("split") == "val")

    if not isinstance(model_params, dict):
        raise ValueError("model_params must be dict")

    if model_name not in REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {list(REGISTRY.keys())}")

    unknown = [n for n in input_names if not is_known_input(n)]
    if unknown:
        raise ValueError(
            f"Unknown input(s): {unknown}. "
            f"Use registered inputs {list(INPUT_REGISTRY)} "
            "or chemprop[path-or-alias]."
        )
    
    if not input_names:
        raise ValueError("At least one feature representation is required")
    
    if len(targets) == 0:
        raise ValueError("Targets are empty")

    required = {"inchikey_full", "inchikey_block", "SMILES", *targets}

    for name, df in [("training", train_df), ("validation", val_df)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} data missing columns: {sorted(missing)}")

    unexpected = set(targets) - set(_SCORED_DEFAULT)
    if unexpected:
        raise ValueError(f"Unexpected targets: {sorted(unexpected)}")

    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate target names")

    for name, df in [("training", train_df), ("validation", val_df)]:
        for column in ("inchikey_full", "inchikey_block"):
            ids = df[column]

            if ids.dtype != pl.String:
                raise ValueError(f"{name}.{column} must contain strings")
            if ids.null_count() > 0:
                raise ValueError(f"{name}.{column} contains nulls")
            if (ids.str.strip_chars() == "").any():
                raise ValueError(f"{name}.{column} contains blank identifiers")

    for name, df in [("training", train_df), ("validation", val_df)]:
        duplicates = (
            df.group_by("inchikey_full")
            .len()
            .filter(pl.col("len") > 1)
        )

        if not duplicates.is_empty():
            raise ValueError(
                f"{name} data contains duplicate molecule IDs: "
                f"{duplicates.head(5).to_dicts()}"
            )    

    overlap = (
    set(train_df["inchikey_block"].to_list())
    & set(val_df["inchikey_block"].to_list())
    )
    if overlap:
        raise ValueError(
            f"Training and validation molecules overlap: {sorted(overlap)[:5]}"
        )

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonblank string")
            
    allowed_targets = set(_SCORED_DEFAULT)
    missing_targets = allowed_targets - set(targets)
    if missing_targets:
        raise ValueError(
            f"Requested targets are missing scored targets: "
            f"{sorted(missing_targets)}"
    )

    for name, df in [("training", train_df), ("validation", val_df)]:
        if df.is_empty():
            raise ValueError(f"{name} data is empty")

    for name, df in [("training", train_df), ("validation", val_df)]:
        smiles = df["SMILES"]

        if smiles.dtype != pl.String:
            raise ValueError(f"{name} SMILES must contain strings")

        if smiles.null_count() > 0:
            raise ValueError(f"{name} SMILES contains nulls")

        if (smiles.str.strip_chars() == "").any():
            raise ValueError(f"{name} SMILES contains blank strings") 

    for name, df in [("training", train_df), ("validation", val_df)]:
        for target in targets:
            labels = df[target]

            if labels.dtype == pl.Null:
                raise ValueError(f"{name}/{target} has no observed labels")

            if not (labels.dtype.is_integer() or labels.dtype.is_float()):
                raise ValueError(f"{name}/{target} labels must be numeric")

            if labels.is_infinite().any():
                raise ValueError(f"{name}/{target} labels contain infinity")

            observed = labels.is_not_null() & labels.is_finite()
            if not observed.any():
                raise ValueError(f"{name}/{target} has no observed labels")

    train_features = featurize(train_df, input_names, cache_dir)
    val_features = featurize(val_df, input_names, cache_dir)

    for name, features, df in [
        ("training", train_features, train_df),
        ("validation", val_features, val_df)
    ]:
        if features.ndim != 2:
            raise ValueError(f"{name} features must be a 2D matrix")

        if features.shape[0] != df.height:
            raise ValueError(f"{name} feature row count does not match molecule count")

        if features.shape[1] == 0:
            raise ValueError(f"{name} features have no columns")

    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError("Training and validation feature widths differ")

    prediction_tables = []
    truth_tables = []

    for target in targets:
        X_train, y_train = drop_nan_rows(
            train_features,
            train_df[target].to_numpy(),
            label=f"train/{target}",
        )

        if len(y_train) == 0:
            raise ValueError(f"No usable training rows for {target}")

        labels = val_df[target]
        observed = (
            labels.is_not_null() & labels.is_finite()
        ).fill_null(False)

        X_val = val_features[observed.to_numpy()]
        observed_val = val_df.filter(observed)
        val_ids = observed_val["inchikey_full"]
        y_val = observed_val[target]

        if len(val_ids) == 0:
            raise ValueError(f"No observed val labels for {target}")

        if not np.isfinite(X_val).all():
            raise ValueError(f"Non-finite val features for {target}")
            
        model = REGISTRY[model_name](**model_params)
        model.fit(X_train, y_train)
        values = np.asarray(model.predict(X_val), dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(f"Predictions for {target} must be one-dimensional")

        if len(values) != len(val_ids):
            raise ValueError(f"Prediction count does not match val rows for {target}")

        if not np.isfinite(values).all():
            raise ValueError(f"Predictions for {target} contain non-finite values")

        target_predictions = pl.DataFrame({
            "model_id": [model_id] * len(val_ids),
            "inchikey_full": val_ids,
            "target": [target] * len(val_ids),
            "pred": values,
        })

        target_truth = pl.DataFrame({
            "inchikey_full": val_ids,
            "target": [target] * len(val_ids),
            "truth": y_val.cast(pl.Float64),
        })

        prediction_tables.append(target_predictions)
        truth_tables.append(target_truth)

    predictions = pl.concat(prediction_tables)
    truth = pl.concat(truth_tables)

    return predictions, truth


def validation_prediction_chemprop(
    df: pl.DataFrame, 
    model_id: str, 
    work_dir: Path, 
    target_columns: list[str],
    epochs: int, 
    seed: int,
    group_column, 
    validation_fraction, 
    validation_seed,
    from_foundation: str | None = None,
    training_timeout: int = 300,
    prediction_timeout: int = 300,
    accelerator: str = "cpu",
    devices: int = 1,
    num_workers: int = 0,
    batch_size: int = 4,
    warmup_epochs: int = 0,
  
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Validation prediction for chemprop only.
    
    Internal validation comes from pinned train, while pinned val is 
    predicted and evaluated.
    """
    
    if "split" not in df.columns:
        raise ValueError("split column must exist")

    train_df = df.filter(pl.col("split") == "train")
    val_df = df.filter(pl.col("split") == "val")

    
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonblank string")
    
    if not target_columns:
        raise ValueError("target_columns must not be empty")

    if any(
        not isinstance(target, str) or not target.strip()
        for target in target_columns
    ):
        raise ValueError("target_columns must contain nonblank strings")

    if len(target_columns) != len(set(target_columns)):
        raise ValueError("target_columns contains duplicate names")

    missing_targets = set(_SCORED_DEFAULT) - set(target_columns)
    if missing_targets:
        raise ValueError(
            f"Missing scored targets: {sorted(missing_targets)}"
        )

    reserved = {"inchikey_full", "inchikey_block", "SMILES", "split", "fold", "model_id"}
    invalid_targets = reserved & set(target_columns)
    if invalid_targets:
        raise ValueError(
            f"Target names conflict with metadata columns: {sorted(invalid_targets)}"
        )

    required = {"inchikey_full", "inchikey_block", "SMILES", *target_columns}

    train_df_missing = required - set(train_df.columns)
    if train_df_missing:
        raise ValueError(f"train_df is missing required columns: {sorted(train_df_missing)}")

    val_df_missing = required - set(val_df.columns)
    if val_df_missing:
        raise ValueError(f"val_df is missing required columns: {sorted(val_df_missing)}")

    for name, df in [
        ("train_df", train_df),
        ("val_df", val_df),
    ]:
        for column in ("inchikey_full", "inchikey_block", "SMILES"):
            values = df[column]
            if values.dtype != pl.String:
                raise ValueError(f"{name}.{column} must contain strings")
            
            if values.null_count() > 0:
                raise ValueError(f"{name}.{column} contains null values")

            if (values.str.strip_chars() == "").any():
                raise ValueError(f"{name}.{column} contains blank values")
            
        duplicates = (
            df.group_by("inchikey_full")
            .len()
            .filter(pl.col("len") > 1)
        )
        if not duplicates.is_empty():
            raise ValueError(f"{name} contains duplicate molecule IDs")

    train_ids = set(train_df["inchikey_block"].to_list())
    val_ids = set(val_df["inchikey_block"].to_list())

    if train_ids & val_ids:
        raise ValueError("Training and validation molecules overlap")

    if len(train_df) == 0:
        raise ValueError("train_df is empty")
    
    if len(val_df) == 0:
        raise ValueError("val_df is empty")

    for name, df in [
        ("train_df", train_df),
        ("val_df", val_df),
    ]:
        for target in target_columns:
            values = df[target]

            # An entirely missing column may have Polars Null dtype.
            if values.dtype != pl.Null and not values.dtype.is_numeric():
                raise ValueError(f"{name}.{target} must be numeric")

            values = values.cast(pl.Float64)

            if values.is_infinite().any():
                raise ValueError(f"{name}.{target} contains infinite values")

    # Standardize dtypes for concatenation.
    # Treat NaN as a missing label and write missing labels as empty CSV cells.
    target_expressions = [
        pl.col(target).cast(pl.Float64).fill_nan(None)
        for target in target_columns
    ]

    train_df = train_df.with_columns(target_expressions)
    val_df = val_df.with_columns(target_expressions)

    for target in _SCORED_DEFAULT:
        if train_df[target].drop_nulls().is_empty():
            raise ValueError(f"No training labels for scored target {target}")

    columns = ["inchikey_full", "SMILES", *target_columns]

    truth_tables = []
    for target in _SCORED_DEFAULT:
        observed_val = val_df.filter(pl.col(target).is_not_null())

        if observed_val.is_empty():
            raise ValueError(f"No validation labels for scored target {target}")

        truth_tables.append(
            observed_val.select(
                "inchikey_full",
                pl.lit(target).alias("target"),
                pl.col(target).cast(pl.Float64).alias("truth"),
            )
        )

    truth = pl.concat(truth_tables)

    fit_df, internal_val_df = _split_internal_validation(
        train_df,
        group_column=group_column,
        validation_fraction=validation_fraction,
        seed=validation_seed,
    )

    training_data = pl.concat([
        fit_df.select(columns).with_columns(
            pl.lit("train").alias("split")
        ),
        internal_val_df.select(columns).with_columns(
            pl.lit("val").alias("split")
        ),
    ])

    prediction_data = val_df.select("inchikey_full", "SMILES")

    work_dir.mkdir(parents=True, exist_ok=True)

    if any(work_dir.iterdir()):
        raise ValueError(
            f"work_dir must be empty for a new training attempt: {work_dir}"
        )

    training_path = work_dir / "training.csv"
    output_dir = work_dir / "chemprop"
    predictions_path = work_dir / "native_predictions.csv"
    val_path = work_dir / "val.csv"

    training_data.write_csv(training_path)
    prediction_data.write_csv(val_path)

    cmd = build_command(
        data_path=training_path,
        smiles_column="SMILES",
        target_columns=target_columns,
        output_dir=output_dir,
        epochs=epochs,
        seed=seed,
        from_foundation=from_foundation,
        extra_args=[
            "--splits-column", "split",
            "--accelerator", accelerator,
            "--devices", str(devices),
            "--num-workers", str(num_workers),
            "--batch-size", str(batch_size),
            "--warmup-epochs", str(warmup_epochs),
        ],
    )

    cmd[0] = str(Path(sys.executable).parent / "chemprop")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=training_timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Chemprop training failed:\n{result.stdout}\n{result.stderr}"
        )

    model_path = output_dir / "model_0" / "best.pt"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Chemprop training did not produce a model at {model_path}"
        )

    predict_cmd = [
        cmd[0],
        "predict",
        "--test-path", str(val_path),
        "--model-path", str(model_path),
        "--preds-path", str(predictions_path),
        "--smiles-columns", "SMILES",
         "--accelerator", accelerator,
        "--devices", str(devices),
        "--num-workers", str(num_workers),
        "--batch-size", str(batch_size),
    ]

    result = subprocess.run(
        predict_cmd,
        capture_output=True,
        text=True,
        timeout=prediction_timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Chemprop prediction failed:\n{result.stdout}\n{result.stderr}"
        )

    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Chemprop did not produce predictions at {predictions_path}"
        )

    predictions = _convert_chemprop_predictions_validation(
        predictions=pl.read_csv(predictions_path),
        truth=truth,
        model_id=model_id,
    )
    
    return predictions, truth


def _convert_chemprop_predictions_validation(
    predictions: pl.DataFrame,
    truth: pl.DataFrame,
    model_id: str, 
) -> pl.DataFrame:
    """Convert and validate chemprop predictions."""

    # Required ID and four scored-target columns exist
    required = {"inchikey_full", *_SCORED_DEFAULT}
    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    duplicates = (predictions.group_by("inchikey_full").len().filter(pl.col("len") > 1))
    if not duplicates.is_empty():
        raise ValueError(
            f"predictions contains duplicate molecules IDs: "
            f"{duplicates.head(5).to_dicts()}" 
        )

    for target in _SCORED_DEFAULT:
        if not predictions[target].dtype.is_numeric():
            raise ValueError(f"Predictions for {target} must be numeric")

    truth_required = {"inchikey_full", "target", "truth"}
    missing = truth_required - set(truth.columns)
    if missing:
        raise ValueError(f"truth is missing required columns: {sorted(missing)}")

    if truth.is_empty():
        raise ValueError("truth table is empty")

    if truth["target"].dtype != pl.String:
        raise ValueError("truth.target must contain strings")

    if any(truth[c].null_count() > 0 for c in truth_required):
        raise ValueError("truth contains null values")

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonblank string")

    if predictions["inchikey_full"].null_count() > 0:
        raise ValueError("predictions.inchikey_full contains nulls")
    
    if predictions["inchikey_full"].dtype != pl.String:
        raise ValueError("predictions.inchikey_full must contain strings")

    if (predictions["inchikey_full"].str.strip_chars() == "").any():
        raise ValueError("predictions.inchikey_full contains blank identifiers")

    if truth["inchikey_full"].dtype != pl.String:
        raise ValueError("truth.inchikey_full must contain strings")

    if (truth["inchikey_full"].str.strip_chars() == "").any():
        raise ValueError("truth.inchikey_full contains blanks identifiers")

    truth_dtype = truth["truth"].dtype
    if not (truth_dtype.is_integer() or truth_dtype.is_float()):
        raise ValueError("truth values must be integers or floats")

    if not np.isfinite(truth["truth"].to_numpy()).all():
        raise ValueError("truth contains non-finite values")

    actual_targets = set(truth["target"].to_list())
    expected_targets = set(_SCORED_DEFAULT)

    missing_targets = expected_targets - actual_targets
    if missing_targets:
        raise ValueError(f"Truth missing scored targets: {sorted(missing_targets)}")

    unexpected_targets = actual_targets - expected_targets
    if unexpected_targets:
        raise ValueError(f"Truth contains unexpected targets: {sorted(unexpected_targets)}")

    duplicate_keys = (
        truth.group_by(["inchikey_full", "target"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate_keys.is_empty():
        raise ValueError("truth contains duplicate molecule/target keys")

    # Conversion preserves the correct ID → target → prediction mapping
    prediction_rows = []

    for target in _SCORED_DEFAULT:
        for molecule, value in predictions.select(
            "inchikey_full", target
        ).iter_rows():
            prediction_rows.append({
                "model_id": model_id, 
                "inchikey_full": molecule,
                "target": target,
                "pred": value,
            })
     
    converted = pl.DataFrame(
        prediction_rows,
        schema={
            "model_id": pl.String,
            "inchikey_full": pl.String,
            "target": pl.String,
            "pred": pl.Float64,
        }
    )

    keys = ["inchikey_full", "target"]
    expected_keys = truth.select(keys)

    missing = expected_keys.join(
        converted.select(keys),
        on=keys,
        how="anti",
    )
    if not missing.is_empty():
        raise ValueError("Missing predictions for validation truth keys")

    converted = converted.join(
        expected_keys, 
        on=keys,
        how="semi"
    )

    if converted["pred"].null_count() > 0:
        raise ValueError("Validation predictions contain null values")

    if not np.isfinite(converted["pred"].to_numpy()).all():
        raise ValueError("Validation predictions contain non-finite values")
        
    return converted


def run_validation_seeds(
    *, 
    df: pl.DataFrame, 
    model_name: str, 
    model_id: str, 
    model_params: dict | None = None,
    input_names: list[str] | None = None,
    output_dir: Path,
    cache_dir: Path | None = None,
    chemprop_config: dict | None = None,
    group_column: str | None = None,
    validation_fraction: float | None = None,
    validation_seed: int | None = None, 
    seeds: list[int],
    targets: list[str]
) -> tuple[dict[int, pl.DataFrame], pl.DataFrame]:
    """Returns per-seed predictions and shared truth, without averaging.

    Feature models must support random state.

    Chemprop requires its config and internal-split settings. targets applies
    to feature based models. Chemprop uses target_columns from its config.
    """

    if len(seeds) < 3:
        raise ValueError("There needs to be at least 3 seeds.")

    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seeds must be integers.")

    if len(seeds) - len(set(seeds)) > 0:
        raise ValueError("seeds must be unique.")

    predictions_by_seed = {}
    first_truth = None

    for seed in seeds:
        chemprop_config_copy = None
        model_params_copy = None
        if model_name == "chemprop":
            if not isinstance(chemprop_config, dict):
                raise ValueError("chemprop_config must be dict")
            chemprop_config_copy = chemprop_config.copy()
            chemprop_config_copy["seed"] = seed
            resolved_config = _resolve_chemprop_config(chemprop_config_copy)

            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError("model_id must be a nonblank string")
            
            seed_model_id = f"{model_id}_seed{seed}"
            seed_output_dir = output_dir / f"seed-{seed}"

            predictions, truth = validation_prediction_chemprop(
                df=df,
                model_id=seed_model_id,
                work_dir=seed_output_dir,
                group_column=group_column,
                validation_fraction=validation_fraction,
                validation_seed=validation_seed,
                **resolved_config,
            )
    
        else: 
            if input_names is None:
                raise ValueError("input_names are required")
            if cache_dir is None:
                raise ValueError("cache_dir is required")
            if model_params is None:
                model_params_copy = {}
            else:
                if not isinstance(model_params, dict):
                    raise ValueError("model_params must be dict")
                model_params_copy = model_params.copy()
            model_params_copy["random_state"] = seed

            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError("model_id must be a nonblank string")

            seed_model_id = f"{model_id}_seed{seed}"

            predictions, truth = validation_prediction(
                df=df,
                model_name=model_name, 
                model_id=seed_model_id, 
                model_params=model_params_copy,
                input_names=input_names,
                targets=targets,
                cache_dir=cache_dir,
            )

        predictions_by_seed[seed] = predictions

        truth = truth.sort(["inchikey_full", "target"])
        if first_truth is None:
            first_truth = truth
        if not truth.equals(first_truth):
            raise ValueError("truth is different for different seeds")

    return predictions_by_seed, first_truth


def _seed_averaging(
    prediction_by_seed: dict[int, pl.DataFrame],
    truth: pl.DataFrame,
    model_id: str,
) -> pl.DataFrame:
    """Average seeds for models that have same configuration and differ only in seed.
    
    Inputs must be runs of the same configuration with distinct training seeds, as 
    supplied by the multi-seed validation runner.
    """

    for seed in prediction_by_seed.keys():
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed keys must be integers")

    if not isinstance(model_id, str):
        raise ValueError("model_id must be a string")

    if model_id.strip() == "":
        raise ValueError("model_id cannot be blank")

    if len(prediction_by_seed) < 3:
        raise ValueError("There must be at least three seed entries")

    if "inchikey_full" not in truth.columns:
        raise ValueError("inchikey_full not in truth.columns")

    if "target" not in truth.columns:
        raise ValueError("target not in truth.columns")

    if len(truth) == 0:
        raise ValueError("truth cannot be empty")

    keys = ["inchikey_full", "target"]
    for key in keys:
        values = truth[key]

        if values.dtype != pl.String:
            raise ValueError(f"{key} must contain strings")
        if values.null_count() > 0:
            raise ValueError(f"{key} cannot be null")
        if (values.str.strip_chars() == "").any():
            raise ValueError(f"{key} cannot be blank")

    truth_counts = truth.group_by(keys).len()
    truth_dup_keys = truth_counts.filter(pl.col("len") > 1)
    if len(truth_dup_keys) > 0:
        raise ValueError(f"There are {len(truth_dup_keys)} duplicates in truth.")
    
    expected_keys = truth.select(keys)

    required_columns = {"model_id", "inchikey_full", "target", "pred"}
    for table in prediction_by_seed.values():
        missing_columns = required_columns - set(table.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        
        if table["model_id"].n_unique() != 1:
            raise ValueError("Each prediction table must contain exactly one model_id.")

        for column in ["model_id", "inchikey_full", "target"]:
            values = table[column]
            if values.dtype != pl.String:
                raise ValueError(f"{column} must contain strings")
            if values.null_count() > 0:
                raise ValueError(f"{column} cannot be null")
            if (values.str.strip_chars() == "").any():
                raise ValueError(f"{column} cannot be blank")

        actual_keys = table.select(keys)
    
        missing = expected_keys.join(actual_keys, on=keys, how="anti")
        extra = actual_keys.join(expected_keys, on=keys, how="anti")
    
        if not missing.is_empty():
            raise ValueError("Seed predictions are missing truth keys")
    
        if not extra.is_empty():
            raise ValueError("Seed predictions contain extra keys")

        table_keys = ["model_id", "inchikey_full", "target"]
        prediction_counts = table.group_by(table_keys).len()
        prediction_dup_keys = prediction_counts.filter(pl.col("len") > 1)
        if len(prediction_dup_keys) > 0:
            raise ValueError(f"There are {len(prediction_dup_keys)} duplicates in prediction.")

        if not table["pred"].dtype.is_numeric():
            raise ValueError("pred must be numeric")
        if table["pred"].null_count() > 0:
            raise ValueError("pred cannot be null")
        if not table["pred"].is_finite().all():
            raise ValueError("pred must be finite")

    combined_predictions = pl.concat(list(prediction_by_seed.values()))

    average_prediction = combined_predictions.group_by(
        pl.col("inchikey_full"), 
        pl.col("target"),
    ).agg(pl.col("pred").mean())

    average_prediction = average_prediction.with_columns(pl.lit(model_id).alias("model_id"))

    return average_prediction.select(
        "model_id", "inchikey_full", "target", "pred"
    ).sort(["inchikey_full", "target"])


def r_squared_filtering(
    seed_averaged_validation_predictions,
    validation_truth
) -> pl.DataFrame:
    """Returns scores and selection decisions, not filtered predictions."""

    targets = validation_truth["target"].unique().to_list()
    
    results = []

    for target in targets:
        model_ids = sorted(
            seed_averaged_validation_predictions
            .filter(pl.col("target") == target)["model_id"]
            .unique()
            .to_list()
        )
        if len(model_ids) == 0:
            raise ValueError(f"No candidate models for {target}")

        truth = validation_truth.filter(pl.col("target") == target)
        if truth["truth"].null_count() > 0:
            raise ValueError("truth contains missing labels")
        if not truth["truth"].is_finite().all():
            raise ValueError("truth must be finite")

        for model_id in model_ids:
            validation_predictions = seed_averaged_validation_predictions.filter(
                (pl.col("target") == target)
                & (pl.col("model_id") == model_id)
            )

            aligned = truth.join(
                validation_predictions.select("inchikey_full", "target", "pred"),
                on=["inchikey_full", "target"],
                how="left",
                validate="1:1",
            )

            if aligned["pred"].null_count() > 0:
                raise ValueError("Missing predictions for validation truth")

            if not aligned["pred"].is_finite().all():
                raise ValueError("Predictions must be finite")

            y = aligned["truth"].to_numpy()
            y_pred = aligned["pred"].to_numpy()

            numerator = np.sum((y - y_pred) ** 2)
            denominator = np.sum((y - y.mean()) ** 2)

            if len(y) < 2 or denominator == 0:
                raise ValueError("R² requires at least two labels with nonzero variance")

            r_squared = 1 - numerator / denominator

            results.append({
                "model_id": model_id, 
                "target": target, 
                "r_squared": r_squared,
                "keep": bool(r_squared >= 0.3),
            })

    return pl.DataFrame(results)


def heldout_rae_filter(
    seed_averaged_validation_predictions: pl.DataFrame,
    validation_truth: pl.DataFrame,
    seed_averaged_oof_predictions: pl.DataFrame, 
    oof_truth: pl.DataFrame,
) -> pl.DataFrame:
    """Return RAE scores and selection decisions.

    Raises ValueError if OOF RAE is zero because the ratio cannot
    be calculated.
    """

    targets = validation_truth["target"].unique().to_list()

    results = []

    for target in targets:
        model_ids = sorted(
            seed_averaged_validation_predictions
            .filter(pl.col("target") == target)["model_id"]
            .unique()
            .to_list()
        )
        if len(model_ids) == 0:
            raise ValueError(f"No candidate models for {target}")

        truth = validation_truth.filter(pl.col("target") == target)
        if truth["truth"].null_count() > 0:
            raise ValueError("truth contains missing labels")
        if not truth["truth"].is_finite().all():
            raise ValueError("truth must be finite")

        target_oof_truth = oof_truth.filter(pl.col("target") == target)
        if target_oof_truth["truth"].null_count() > 0:
            raise ValueError("oof_truth contains missing labels")
        if not target_oof_truth["truth"].is_finite().all():
            raise ValueError("oof_truth must be finite")

        for model_id in model_ids:
            validation_predictions = seed_averaged_validation_predictions.filter(
                (pl.col("target") == target)
                & (pl.col("model_id") == model_id)
            )

            validation_aligned = truth.join(
                validation_predictions.select("inchikey_full", "target", "pred"),
                on=["inchikey_full", "target"],
                how="left",
                validate="1:1",
            )

            if validation_aligned["pred"].null_count() > 0:
                raise ValueError("Missing predictions for validation truth")

            if not validation_aligned["pred"].is_finite().all():
                raise ValueError("Predictions must be finite")

            heldout_y = validation_aligned["truth"].to_numpy()
            heldout_y_pred = validation_aligned["pred"].to_numpy()

            heldout_numerator = np.sum(abs(heldout_y - heldout_y_pred))
            heldout_denominator = np.sum(abs(heldout_y - heldout_y.mean()))

            if len(heldout_y) < 2 or heldout_denominator == 0:
                raise ValueError("heldout_rae requires at least two labels with nonzero variance")

            heldout_rae = heldout_numerator / heldout_denominator

            oof_predictions = seed_averaged_oof_predictions.filter(
                (pl.col("target") == target)
                & (pl.col("model_id") == model_id)
            )

            oof_aligned = target_oof_truth.join(
                oof_predictions.select("inchikey_full", "fold", "target", "pred"),
                on=["inchikey_full", "fold", "target"],
                how="left",
                validate="1:1",
            )

            if oof_aligned["pred"].null_count() > 0:
                raise ValueError("Missing predictions for oof truth")

            if not oof_aligned["pred"].is_finite().all():
                raise ValueError("Predictions must be finite")

            oof_y = oof_aligned["truth"].to_numpy()
            oof_y_pred = oof_aligned["pred"].to_numpy()

            oof_numerator = np.sum(abs(oof_y - oof_y_pred))
            oof_denominator = np.sum(abs(oof_y - oof_y.mean()))

            if len(oof_y) < 2 or oof_denominator == 0:
                raise ValueError("oof_rae requires at least two labels with nonzero variance")

            oof_rae = oof_numerator / oof_denominator

            if oof_rae == 0:
                raise ValueError("oof_rae is 0.")

            rae_ratio = heldout_rae / oof_rae

            results.append({
                "model_id": model_id, 
                "target": target, 
                "heldout_rae": heldout_rae,
                "oof_rae": oof_rae,
                "rae_ratio": rae_ratio,
                "keep": bool(rae_ratio <= 2.0),
            })

    return pl.DataFrame(results)


def deduplication_filter(
    seed_averaged_validation_predictions: pl.DataFrame,
    validation_truth: pl.DataFrame,
    seed_averaged_oof_predictions: pl.DataFrame, 
    oof_truth: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Cluster models with highly correlated validation predictions per target.

    Average each cluster's predictions in both validation and OOF populations
    using the same membership. Raise ValueError if correlations are undefined.
    Return clustered validation predictions, OOF predictions, and membership.
    """

    targets = validation_truth["target"].unique().to_list()

    validation_tables = []
    oof_tables = []
    membership_rows = []

    for target in targets:
        model_ids = sorted(
            seed_averaged_validation_predictions
            .filter(pl.col("target") == target)["model_id"]
            .unique()
            .to_list()
        )
        if len(model_ids) == 0:
            raise ValueError("No surviving models for target")

        truth = validation_truth.filter(pl.col("target") == target)
        if truth["truth"].null_count() > 0:
            raise ValueError("truth contains missing labels")
        if not truth["truth"].is_finite().all():
            raise ValueError("truth must be finite")

        target_oof_truth = oof_truth.filter(pl.col("target") == target)
        if target_oof_truth["truth"].null_count() > 0:
            raise ValueError("oof_truth contains missing labels")
        if not target_oof_truth["truth"].is_finite().all():
            raise ValueError("oof_truth must be finite")

        validation_keys = truth.select(
            "inchikey_full", "target"
        ).sort("inchikey_full")

        oof_keys = target_oof_truth.select(
            "inchikey_full", "fold", "target"
        ).sort(["inchikey_full", "fold"])

        validation_columns = []
        oof_columns = []

        for model_id in model_ids:
            validation_predictions = seed_averaged_validation_predictions.filter(
                (pl.col("target") == target)
                & (pl.col("model_id") == model_id)
            )

            validation_aligned = truth.join(
                validation_predictions.select("inchikey_full", "target", "pred"),
                on=["inchikey_full", "target"],
                how="left",
                validate="1:1",
            )

            if validation_aligned["pred"].null_count() > 0:
                raise ValueError("Missing predictions for validation truth")

            if not validation_aligned["pred"].is_finite().all():
                raise ValueError("Predictions must be finite")

            oof_predictions = seed_averaged_oof_predictions.filter(
                (pl.col("target") == target)
                & (pl.col("model_id") == model_id)
            )

            oof_aligned = target_oof_truth.join(
                oof_predictions.select("inchikey_full", "fold", "target", "pred"),
                on=["inchikey_full", "fold", "target"],
                how="left",
                validate="1:1",
            )

            if oof_aligned["pred"].null_count() > 0:
                raise ValueError("Missing predictions for oof truth")

            if not oof_aligned["pred"].is_finite().all():
                raise ValueError("Predictions must be finite")

            validation_columns.append(validation_aligned.sort("inchikey_full")["pred"].to_numpy())
            oof_columns.append(oof_aligned.sort(["inchikey_full", "fold"])["pred"].to_numpy())

        X_validation = np.column_stack(validation_columns)
        X_oof = np.column_stack(oof_columns)

        if len(model_ids) == 1:
            cluster_labels = np.array([1])
        else:
            correlations = np.corrcoef(X_validation, rowvar=False)

            if not np.isfinite(correlations).all():
                 raise ValueError("Cannot cluster undefined correlations")

            distances = np.clip(1.0 - correlations, 0.0, 2.0)
            distances = (distances + distances.T) / 2
            np.fill_diagonal(distances, 0.0)

            condensed = squareform(distances)
            tree = linkage(condensed, method="complete")

            cutoff = np.nextafter(1.0 - 0.95, -np.inf)
            cluster_labels = fcluster(tree, t=cutoff, criterion="distance")

        for cluster in np.unique(cluster_labels):
            members = np.flatnonzero(cluster_labels == cluster)

            validation_mean = X_validation[:, members].mean(axis=1)
            oof_mean = X_oof[:, members].mean(axis=1)

            cluster_id = f"{target}_cluster_{int(cluster)}"

            for member in members:
                membership_rows.append({
                    "target": target,
                    "cluster_id": cluster_id,
                    "model_id": model_ids[member],
                })

            validation_tables.append(
                validation_keys.with_columns(
                    pl.lit(cluster_id).alias("model_id"),
                    pl.Series("pred", validation_mean),
                ).select("model_id", "inchikey_full", "target", "pred")
            )

            oof_tables.append(
                oof_keys.with_columns(
                    pl.lit(cluster_id).alias("model_id"),
                    pl.Series("pred", oof_mean),
                ).select("model_id", "inchikey_full", "fold", "target", "pred")
            )

    return pl.concat(validation_tables), pl.concat(oof_tables), pl.DataFrame(membership_rows),



                
            
