"""Generate, validate, and save OOF predictions using saved fold assignments.

Use full InChIKeys for row identity and reject connectivity blocks that
span folds. Support feature models, Chemprop, and multiple training seeds.
"""

import argparse
import hashlib
import json
import math
import tempfile
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl

from data_tools.inputs import INPUT_REGISTRY, featurize, is_known_input
from models import REGISTRY
from models.utils import drop_nan_rows
from representations.chemprop_weights import parse_spec


_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

_COMPLETION_VERSION = 1


def _validate_inputs(truth: pl.DataFrame, heldout_fold: int, predictions: pl.DataFrame, model_id: str) -> None:
    """
    truth already contains the saved fold assignments, the function validates
    one model/fold job, and failures raise ValueError.

    Requires the requested held-out fold to contain all four scored targets;
    individual molecules may have labels for only a subset of them.

    Validates the entire supplied truth table for duplicates and fold consistency.
    """
    # Required columns
    prediction_required = {"model_id", "inchikey_full", "fold", "target", "pred"}
    prediction_missing = prediction_required - set(predictions.columns)
    if prediction_missing:
        raise ValueError(f"predictions is missing required columns: {sorted(prediction_missing)}")
        
    truth_required = {"inchikey_full", "fold", "target", "truth"}
    truth_missing = truth_required - set(truth.columns)
    if truth_missing:
        raise ValueError(f"truth is missing required columns: {sorted(truth_missing)}")

    # Empty tables, nulls, types, blank
    if len(predictions) == 0:
        raise ValueError("predictions table is empty")
    if len(truth) == 0:
        raise ValueError("truth table is empty")
    
    prediction_null_columns = [
        c for c in prediction_required
        if predictions[c].null_count() > 0
    ]
    if prediction_null_columns:
        raise ValueError(f"There are null values in predictions: {sorted(prediction_null_columns)}")
        
    truth_null_columns = [
        c for c in truth_required
        if truth[c].null_count() > 0
    ]
    if truth_null_columns:
        raise ValueError(f"There are null values in truth: {sorted(truth_null_columns)}")
    
    if predictions["model_id"].dtype != pl.String:
        raise ValueError("predictions.model_id must contain strings")

    if predictions["inchikey_full"].dtype != pl.String:
        raise ValueError("predictions.inchikey_full must contain strings")

    if truth["inchikey_full"].dtype != pl.String:
        raise ValueError("truth.inchikey_full must contain strings")

    if predictions["target"].dtype != pl.String:
        raise ValueError("predictions.target must contain strings")

    if truth["target"].dtype != pl.String:
        raise ValueError("truth.target must contain strings")

    if not predictions["fold"].dtype.is_integer():
        raise ValueError("predictions.fold must contain integers")

    if not truth["fold"].dtype.is_integer():
        raise ValueError("truth.fold must contain integers")

    if (predictions["model_id"].str.strip_chars() == "").any():
        raise ValueError("predictions.model_id contains blank identifiers")

    if (predictions["inchikey_full"].str.strip_chars() == "").any():
        raise ValueError("predictions.inchikey_full contains blank identifiers")
    
    if (truth["inchikey_full"].str.strip_chars() == "").any():
        raise ValueError("truth.inchikey_full contains blank identifiers")
        
    if predictions["pred"].dtype != pl.Float64:
        raise ValueError("pred must contain Float64 values")
    
    if truth["truth"].dtype != pl.Float64:
        raise ValueError("truth must contain Float64 values")

    if not np.isfinite(predictions["pred"].to_numpy()).all():
        raise ValueError("pred contains non-finite values")

    if not np.isfinite(truth["truth"].to_numpy()).all():
        raise ValueError("truth contains non-finite values")

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonblank string")

    if isinstance(heldout_fold, bool) or not isinstance(heldout_fold, int):
        raise ValueError("heldout_fold must be an integer")

    # Select expected fold and reject if empty
    key_columns = ["inchikey_full", "fold", "target"]

    expected_keys = truth.filter(
        pl.col("fold") == heldout_fold
    ).select(key_columns)

    if expected_keys.is_empty():
        raise ValueError(f"No truth rows for held-out fold {heldout_fold}")
    
    actual_keys = predictions.select(key_columns)

    # Duplicate keys
    prediction_keys = ["model_id", "inchikey_full", "fold", "target"]
    prediction_counts = predictions.group_by(prediction_keys).len()
    prediction_dup_keys = prediction_counts.filter(pl.col("len") > 1)
    if len(prediction_dup_keys) > 0:
        raise ValueError(f"There are {len(prediction_dup_keys)} duplicates in prediction.")
    
    truth_keys_columns = ["inchikey_full", "fold", "target"]
    truth_counts = truth.group_by(truth_keys_columns).len()
    truth_dup_keys = truth_counts.filter(pl.col("len") > 1)
    if len(truth_dup_keys) > 0:
        raise ValueError(f"There are {len(truth_dup_keys)} duplicates in truth.")

    # Expected model ID
    wrong_model = predictions.filter(
        (pl.col("model_id") != model_id)
    ) 

    if len(wrong_model) > 0:
        raise ValueError(
            f"Expected all predictions to belong to model {model_id!r}. "
            f"Found {len(wrong_model)} invalid row(s): "
            f"{wrong_model.head(5).to_dicts()}"
        )

    # Missing/extra key coverage
    missing = expected_keys.join(actual_keys, on=key_columns, how="anti")
    if len(missing) > 0: 
        raise ValueError(
            f"missing required prediction key(s): "
            f"{missing.head(5).to_dicts()}"
        )
    extra = actual_keys.join(expected_keys, on=key_columns, how="anti")
    if len(extra) > 0: 
        raise ValueError(
            f"unexpected prediction key(s): "
            f"{extra.head(5).to_dicts()}"
        )

    # Truth fold consistency
    truth_fold_counts = truth.group_by("inchikey_full").agg(
        pl.col("fold").n_unique().alias("n_folds")
    )
    inconsistent_truth = truth_fold_counts.filter(pl.col("n_folds") > 1)
    if len(inconsistent_truth) > 0:
        raise ValueError(
            f"Truth molecules assigned to multiple folds: "
            f"{inconsistent_truth.head(5).to_dicts()}"
        )

    allowed_targets = set(_SCORED_DEFAULT)

    # Reject unexpected targets anywhere in the supplied truth.
    truth_targets = set(truth["target"].to_list())
    unexpected_truth = truth_targets - allowed_targets
    if unexpected_truth:
        raise ValueError(
            f"Truth contains unexpected targets: {sorted(unexpected_truth)}"
        )

    prediction_targets = set(predictions["target"].to_list())
    unexpected_predictions = prediction_targets - allowed_targets
    if unexpected_predictions:
        raise ValueError(
            f"Predictions contain unexpected targets: {sorted(unexpected_predictions)}"
        )

    # Require all four scored targets in this held-out fold.
    fold_targets = set(expected_keys["target"].to_list())
    missing_targets = allowed_targets - fold_targets
    if missing_targets:
        raise ValueError(
            f"Fold {heldout_fold} is missing scored targets: "
            f"{sorted(missing_targets)}"
        )


def _make_run_id(config: dict) -> str:
    """Return a deterministic hash of the resolved run configuration."""
    serialized = json.dumps(
        config,
        sort_keys=True, # Makes dictionary insertion order irrelevant
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file's contents."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def _validate_completion(
        job_dir: Path, 
        truth: pl.DataFrame, 
        run_id: str, 
        model_id: str, 
        heldout_fold: int,
) -> bool:
    """Return whether saved artifacts form a valid completed job."""

    prediction_path = job_dir / "predictions.csv"
    completion_path = job_dir / "complete.json"

    # Missing files, malformed JSON as incomplete jobs
    try:
        completion = json.loads(
            completion_path.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    
    try:
        predictions = pl.read_csv(prediction_path)
    except (
        FileNotFoundError,
        pl.exceptions.NoDataError,
        pl.exceptions.ComputeError,
    ):
        return False

    if not isinstance(completion, dict):
        return False

    version = completion.get("contract_version")
    if type(version) is not int or version != _COMPLETION_VERSION:
        return False

    if completion.get("run_id") != run_id:
        return False

    if completion.get("model_id") != model_id:
        return False

    fold = completion.get("fold")
    if type(fold) is not int or fold != heldout_fold:
        return False

    n_predictions = completion.get("n_predictions")
    if type(n_predictions) is not int or n_predictions != predictions.height:
        return False

    if completion.get("prediction_sha256") != _file_sha256(prediction_path):
        return False

    try:
        _validate_inputs(
            truth=truth, 
            heldout_fold=heldout_fold,
            predictions=predictions,
            model_id=model_id,
        )
    except ValueError:
        return False

    return True
    

def _write_fold_artifacts(
        job_dir: Path, 
        predictions: pl.DataFrame, 
        truth: pl.DataFrame,
        run_id: str, 
        model_id: str, 
        heldout_fold: int,
) -> None:
    """
    Construct completion metadata using the saved CSV’s checksum.
    Write temporary JSON, then rename it to complete.json last.
    """
    _validate_inputs(truth, heldout_fold, predictions, model_id)

    job_dir.mkdir(parents=True, exist_ok=True)

    temporary_path = job_dir / "predictions.tmp.csv"
    prediction_path = job_dir / "predictions.csv"

    predictions.write_csv(temporary_path)

    saved_predictions = pl.read_csv(temporary_path)

    _validate_inputs(
        truth=truth,
        heldout_fold=heldout_fold,
        predictions=saved_predictions,
        model_id=model_id,
    )

    temporary_path.replace(prediction_path)

    completion = {
        "contract_version": _COMPLETION_VERSION,
        "run_id": run_id, 
        "model_id": model_id,
        "fold": heldout_fold, 
        "prediction_sha256": _file_sha256(prediction_path),
        "n_predictions": saved_predictions.height,
    }

    temporary_json = json.dumps(
            completion,
            sort_keys=True, # makes dictionary insertion order irrelevant
            separators=(",", ":"),
            allow_nan=False,
        )

    temporary_completion_path = job_dir / "complete.tmp.json"
    completion_path = job_dir / "complete.json"

    temporary_completion_path.write_text(
        temporary_json,
        encoding="utf-8",
    )
    temporary_completion_path.replace(completion_path)

    
def _run_fold_job(
    job_dir: Path,
    truth: pl.DataFrame,
    run_id: str, 
    model_id: str, 
    heldout_fold: int, 
    generate_predictions: Callable[[], pl.DataFrame],
) -> bool:
    """Return True if the job ran, or False if valid output was reused."""

    if _validate_completion(
        job_dir=job_dir,
        truth=truth,
        run_id=run_id,
        model_id=model_id,
        heldout_fold=heldout_fold,
    ):
        return False
    
    predictions = generate_predictions()

    _write_fold_artifacts(
        job_dir=job_dir, 
        predictions=predictions,
        truth=truth,
        run_id=run_id,
        model_id=model_id,
        heldout_fold=heldout_fold,
    )

    return True


def _generate_fold_predictions(
    train_df: pl.DataFrame,
    heldout_df: pl.DataFrame,
    model_name: str, 
    model_id: str, 
    model_params: dict,
    input_names: list[str],
    targets: list[str],
    heldout_fold: int,
    cache_dir: Path,  
) -> pl.DataFrame:
    """Fit per-target models and return long-format predictions for one held-out fold.

    For each target, train on finite labeled rows and predict labeled held-out rows.
    """

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

    for name, df in [("training", train_df), ("held-out", heldout_df)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} data missing columns: {sorted(missing)}")

    unexpected = set(targets) - set(_SCORED_DEFAULT)
    if unexpected:
        raise ValueError(f"Unexpected targets: {sorted(unexpected)}")

    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate target names")

    for name, df in [("training", train_df), ("held-out", heldout_df)]:
        for column in ["inchikey_full", "inchikey_block"]:
            ids = df[column]

            if ids.dtype != pl.String:
                raise ValueError(f"{name}.{column} must contain strings")

            if ids.null_count() > 0:
                raise ValueError(f"{name}.{column} contains nulls")

            if (ids.str.strip_chars() == "").any():
                raise ValueError(f"{name}.{column} contains blank identifiers")

    for name, df in [("training", train_df), ("held-out", heldout_df)]:
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
    & set(heldout_df["inchikey_block"].to_list())
    )
    if overlap:
        raise ValueError(
            f"Training and held-out molecules overlap: {sorted(overlap)[:5]}"
        )

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonblank string")

    if isinstance(heldout_fold, bool) or not isinstance(heldout_fold, int):
        raise ValueError("heldout_fold must be an integer")
            
    allowed_targets = set(_SCORED_DEFAULT)
    missing_targets = allowed_targets - set(targets)
    if missing_targets:
        raise ValueError(
            f"Requested targets are missing scored targets: "
            f"{sorted(missing_targets)}"
    )

    for name, df in [("training", train_df), ("held-out", heldout_df)]:
        if df.is_empty():
            raise ValueError(f"{name} data is empty")

    for name, df in [("training", train_df), ("held-out", heldout_df)]:
        smiles = df["SMILES"]

        if smiles.dtype != pl.String:
            raise ValueError(f"{name} SMILES must contain strings")

        if smiles.null_count() > 0:
            raise ValueError(f"{name} SMILES contains nulls")

        if (smiles.str.strip_chars() == "").any():
            raise ValueError(f"{name} SMILES contains blank strings") 

    for name, df in [("training", train_df), ("held-out", heldout_df)]:
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
    heldout_features = featurize(heldout_df, input_names, cache_dir)

    for name, features, df in [
        ("training", train_features, train_df),
        ("held-out", heldout_features, heldout_df)
    ]:
        if features.ndim != 2:
            raise ValueError(f"{name} features must be a 2D matrix")

        if features.shape[0] != df.height:
            raise ValueError(f"{name} feature row count does not match molecule count")

        if features.shape[1] == 0:
            raise ValueError(f"{name} features have no columns")

    if train_features.shape[1] != heldout_features.shape[1]:
        raise ValueError("Training and held-out feature widths differ")

    prediction_tables = []

    for target in targets:
        X_train, y_train = drop_nan_rows(
            train_features,
            train_df[target].to_numpy(),
            label=f"train/{target}",
        )

        if len(y_train) == 0:
            raise ValueError(f"No usable training rows for {target}")

        labels = heldout_df[target]
        observed = (
            labels.is_not_null() & labels.is_finite()
        ).fill_null(False)

        X_heldout = heldout_features[observed.to_numpy()]
        heldout_ids = heldout_df.filter(observed)["inchikey_full"]

        if len(heldout_ids) == 0:
            raise ValueError(f"No observed held-out labels for {target}")

        if not np.isfinite(X_heldout).all():
            raise ValueError(f"Non-finite held-out features for {target}")
        
        model = REGISTRY[model_name](**model_params)
        model.fit(X_train, y_train)
        values = np.asarray(model.predict(X_heldout), dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(f"Predictions for {target} must be one-dimensional")

        if len(values) != len(heldout_ids):
            raise ValueError(f"Prediction count does not match held-out rows for {target}")

        if not np.isfinite(values).all():
            raise ValueError(f"Predictions for {target} contain non-finite values")

        target_predictions = pl.DataFrame({
            "model_id": [model_id] * len(heldout_ids),
            "inchikey_full": heldout_ids,
            "fold": [heldout_fold] * len(heldout_ids),
            "target": [target] * len(heldout_ids),
            "pred": values,
        })

        prediction_tables.append(target_predictions)

    return pl.concat(prediction_tables)


def _load_oof_data(data_path: Path, folds_path: Path) -> pl.DataFrame:
    """Load and validate molecule data and fold assignments, then join them.
    
    data_path must already contain only the approved OOF training population

    Note: Using both inchikey_full and inchikey_block. inchikey_full is used
    for row uniqueness, fold_coverage, and join. inchikey_block is used to 
    check that records sharing a skeleton aren't assigned to different folds. 
    """

    data = pl.read_csv(data_path, infer_schema_length=None)
    folds = pl.read_csv(folds_path, infer_schema_length=None)

    data_required = {
        "inchikey_full",
        "inchikey_block",
        "SMILES",
        *_SCORED_DEFAULT,
    }
    data_missing = data_required - set(data.columns)
    if data_missing:
        raise ValueError(f"data is missing required columns: {sorted(data_missing)}")

    if "fold" in data.columns:
        raise ValueError("data already contains a fold column")

    folds_required = {"inchikey_full", "fold"}
    folds_missing = folds_required - set(folds.columns)
    if folds_missing:
        raise ValueError(f"folds is missing required columns: {sorted(folds_missing)}")
    
    if len(data) == 0:
        raise ValueError("data table is empty")
    if len(folds) == 0:
        raise ValueError("folds is empty")

    if data["inchikey_full"].dtype != pl.String:
        raise ValueError("data.inchikey_full must contain strings")
    if data["inchikey_block"].dtype != pl.String:
        raise ValueError("data.inchikey_block must contain strings")
    if data["SMILES"].dtype != pl.String:
        raise ValueError("data.SMILES must contain strings")
    for target in _SCORED_DEFAULT:
        if not data[target].dtype.is_numeric():
            raise ValueError(f"data.{target} must contain numeric labels")

    if folds["inchikey_full"].dtype != pl.String:
        raise ValueError("folds.inchikey_full must contain strings")
    if not folds["fold"].dtype.is_integer():
        raise ValueError(f"folds.fold must contain integers")

    if (data["inchikey_full"].str.strip_chars() == "").any():
        raise ValueError("data.inchikey_full contains blank identifiers")
    if (data["inchikey_block"].str.strip_chars() == "").any():
        raise ValueError("data.inchikey_block contains blank identifiers")
    if (data["SMILES"].str.strip_chars() == "").any():
        raise ValueError("data.SMILES contains blank identifiers")

    if (folds["inchikey_full"].str.strip_chars() == "").any():
        raise ValueError("folds.inchikey_full contains blank identifiers")

    data_null_columns = [
        c for c in ["inchikey_full", "inchikey_block", "SMILES"]
        if data[c].null_count() > 0
    ]
    if data_null_columns:
        raise ValueError(f"There are null values in data: {sorted(data_null_columns)}")

    folds_null_columns = [
        c for c in folds_required
        if folds[c].null_count() > 0
    ]
    if folds_null_columns:
        raise ValueError(f"There are null values in folds: {sorted(folds_null_columns)}")

    if folds["fold"].n_unique() < 2:
        raise ValueError("At least two distinct folds are required")
        
    for name, table in [("data", data), ("folds", folds)]:
        duplicates = (
            table.group_by("inchikey_full")
            .len()
            .filter(pl.col("len") > 1)
        )
        if not duplicates.is_empty():
            raise ValueError(
                f"{name} contains duplicate molecules IDs: "
                f"{duplicates.head(5).to_dicts()}" 
            )
   
    data_ids = data.select("inchikey_full")
    fold_ids = folds.select("inchikey_full")

    missing = data_ids.join(fold_ids, on="inchikey_full", how="anti")
    extra = fold_ids.join(data_ids, on="inchikey_full", how="anti")

    if not missing.is_empty():
        raise ValueError(f"Molecules missing fold assignments: {missing.head(5).to_dicts()}")

    if not extra.is_empty():
        raise ValueError(f"Assignments reference unknown molecules: {extra.head(5).to_dicts()}")

    if (folds["fold"] < 0).any():
        raise ValueError("Fold assignments must be nonnegative")
    
    joined = data.join(
        folds.select("inchikey_full", "fold"), 
        on="inchikey_full", 
        how="left",
        validate="1:1",
    )

    block_folds = joined.group_by("inchikey_block").agg(
        pl.col("fold").n_unique().alias("n_folds")
    )
    if (block_folds["n_folds"] > 1).any():
        raise ValueError("Molecules sharing an inchikey_block must share a fold")

    return joined


def run_oof(
    *, 
    data_path: Path,
    folds_path: Path, 
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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate OOF predictions using saved folds within pinned training data.

    Data requires inchikey_full, inchikey_block, SMILES, and scored targets.
    Fold assignments require inchikey_full and fold. Full keys identify rows;
    blocks keep related molecules together.

    Returns predictions (model_id, inchikey_full, fold, target, pred) and
    truth (inchikey_full, fold, target, truth).
    """

    data = _load_oof_data(data_path, folds_path)

    if model_name == "chemprop":
        from models.oof_chemprop import _resolve_chemprop_config

        resolved_config = _resolve_chemprop_config(chemprop_config)

        if not isinstance(group_column, str) or not group_column.strip():
            raise ValueError("Chemprop requires a nonblank group column")

        if group_column not in data.columns:
            raise ValueError(f"Missing group column: {group_column}")

        if isinstance(validation_seed, bool) or not isinstance(validation_seed, int):
            raise ValueError("validation_seed must be integer")

        if (
            isinstance(validation_fraction, bool)
            or not isinstance(validation_fraction, (int, float))
            or not math.isfinite(validation_fraction)
            or not 0 < validation_fraction < 1
        ):
            raise ValueError(
                "validation_fraction must be a finite number strictly between 0 and 1"
            )

    else:
        if model_name not in REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        if not input_names:
            raise ValueError(f"input_names is required for feature-based models")

        if cache_dir is None:
            raise ValueError("cache_dir is required for feature-based models")

    truth_rows = []

    for target in _SCORED_DEFAULT:
        for molecule, fold, value in data.select(
            "inchikey_full", "fold", target
        ).iter_rows():
            if value is None or np.isnan(value):
                continue
            if np.isinf(value):
                raise ValueError(f"{target} contains infinite labels")

            truth_rows.append({
                "inchikey_full": molecule, 
                "fold": fold,
                "target": target, 
                "truth": float(value),
            })

    truth = pl.DataFrame(
        truth_rows, 
        schema={
            "inchikey_full": pl.String, 
            "fold": pl.Int64,
            "target": pl.String,
            "truth": pl.Float64,
        },
    )

    if truth.is_empty():
        raise ValueError("No observed scored labels")

    if model_name == "chemprop":
        foundation = resolved_config["from_foundation"]

        if foundation is None:
            foundation_identity = None
        elif foundation == "chemeleon":
            checkpoint = Path.home() / ".chemprop" / "chemeleon_mp.pt"

            if not checkpoint.is_file():
                raise FileNotFoundError(
                    "CheMeleon weights must be downloaded before creating "
                    "the OOF resume fingerprint."
                )

            foundation_identity = {
                "name": "chemeleon",
                "sha256": _file_sha256(checkpoint),
            }
            
        else:
            foundation_identity = {
                "sha256": _file_sha256(Path(foundation))
            }

        chemprop_version = version("chemprop")
    
        fingerprint = {
            "data_sha256": _file_sha256(data_path),
            "folds_sha256": _file_sha256(folds_path),
            "model_name": model_name, 
            "model_id": model_id,
            "chemprop_config": {
                key: value
                for key, value in resolved_config.items()
                if key not in {"training_timeout", "prediction_timeout"}
            },
            "internal_validation": {
                "policy": "shuffled_group_prefix_v1",
                "group_column": group_column,
                "fraction": validation_fraction,
                "seed": validation_seed,
            },
            "foundation_identity": foundation_identity,
            "chemprop_version": chemprop_version,
            "adapter_version": 1,
        }
    
        run_id = _make_run_id(fingerprint)

    else:
        if model_params is None:
            model_params = {}

        if not isinstance(model_params, dict):
            raise ValueError("model_params must be a dictionary")

        checkpoint_identities = {}

        for name in input_names:
            checkpoint = parse_spec(name)
            if checkpoint is not None:
                checkpoint = checkpoint.resolve()
                checkpoint_identities[name] = {
                    "path": str(checkpoint),
                    "sha256": _file_sha256(checkpoint),
                }

        run_id = _make_run_id({
            "data_sha256": _file_sha256(data_path),
            "folds_sha256": _file_sha256(folds_path),
            "model_name": model_name,
            "model_id": model_id,
            "model_params": model_params,
            "input_names": input_names,
            "checkpoint_identities": checkpoint_identities,
            "targets": _SCORED_DEFAULT,
        })

    fold_predictions = []

    folds = data["fold"].unique().sort().to_list()

    for fold in folds:
        train_df = data.filter(pl.col("fold") != fold)
        heldout_df = data.filter(pl.col("fold") == fold)

        job_dir = output_dir / f"fold-{fold}"

        def generate_predictions():
            if model_name == "chemprop":
                # Need to clean this up soon
                from models.oof_chemprop import _split_internal_validation, _generate_chemprop_fold_predictions
                train_df_chem, val_df =  _split_internal_validation(
                    training_pool=train_df,
                    group_column=group_column,
                    validation_fraction=validation_fraction,
                    seed=validation_seed,
                )

                job_dir.mkdir(parents=True, exist_ok=True)
                work_dir = Path(tempfile.mkdtemp(prefix="attempt-", dir=job_dir))

                return _generate_chemprop_fold_predictions(
                    train_df=train_df_chem,
                    val_df=val_df,
                    heldout_df=heldout_df,
                    truth=truth,
                    model_id=model_id,
                    heldout_fold=fold,
                    work_dir=work_dir,
                    **resolved_config,
                )

            else: 
                return _generate_fold_predictions(
                    train_df=train_df,
                    heldout_df=heldout_df,
                    model_name=model_name,
                    model_id=model_id,
                    model_params=model_params,
                    input_names=input_names,
                    targets=_SCORED_DEFAULT,
                    heldout_fold=fold,
                    cache_dir=cache_dir,
                )

        _run_fold_job(
            job_dir=job_dir,
            truth=truth,
            run_id=run_id,
            model_id=model_id,
            heldout_fold=fold,
            generate_predictions=generate_predictions,
        )

        fold_predictions.append(
            pl.read_csv(job_dir / "predictions.csv")
        )

    predictions = pl.concat(fold_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)

    temporary_predictions = output_dir / "oof_predictions.tmp.csv"
    temporary_truth = output_dir / "oof_truth.tmp.csv"

    predictions.write_csv(temporary_predictions)
    truth.write_csv(temporary_truth)

    temporary_predictions.replace(output_dir / "oof_predictions.csv")
    temporary_truth.replace(output_dir / "truth.csv")

    return predictions, truth


def run_oof_seeds(
    *, 
    data_path: Path,
    folds_path: Path, 
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
) -> tuple[dict[int, pl.DataFrame], pl.DataFrame]:
    """Call run_oof for multiple seeds.
    
    This wrapper only supports Chemprop and feature-based models which
    accept random_state. 
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
        else: 
            if model_params is None:
                model_params_copy = {}
            else:
                if not isinstance(model_params, dict):
                    raise ValueError("model_params must be dict")
                model_params_copy = model_params.copy()
            model_params_copy["random_state"] = seed

        seed_model_id = f"{model_id}_seed{seed}"
        seed_output_dir = output_dir / f"seed-{seed}"

        predictions, truth = run_oof(
            data_path=data_path,
            folds_path=folds_path, 
            model_name=model_name, 
            model_id=seed_model_id, 
            model_params=model_params_copy,
            input_names=input_names,
            output_dir=seed_output_dir,
            cache_dir=cache_dir,
            chemprop_config=chemprop_config_copy,
            group_column=group_column,
            validation_fraction=validation_fraction,
            validation_seed=validation_seed, 
        )
        predictions_by_seed[seed] = predictions

        if first_truth is None:
            first_truth = truth
        if not truth.equals(first_truth):
            raise ValueError("truth is different for different seeds")

    return predictions_by_seed, first_truth


def _convert_chemprop_predictions(
    predictions: pl.DataFrame,
    truth: pl.DataFrame,
    model_id: str, 
    heldout_fold: int,
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

    truth_required = {"inchikey_full", "fold", "target", "truth"}
    missing = truth_required - set(truth.columns)
    if missing:
        raise ValueError(f"truth is missing required columns: {sorted(missing)}")

    if truth.is_empty():
        raise ValueError("truth table is empty")

    if not truth["fold"].dtype.is_integer():
        raise ValueError("truth.fold must contain integers")

    if truth["target"].dtype != pl.String:
        raise ValueError("truth.target must contain strings")

    if any(truth[c].null_count() > 0 for c in truth_required):
        raise ValueError("truth contains null values")

    if isinstance(heldout_fold, bool) or not isinstance(heldout_fold, int):
        raise ValueError("heldout_fold must be an integer")

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

    truth = truth.with_columns(
        pl.col("fold").cast(pl.Int64)
    )

    expected_ids = truth.filter(
        pl.col("fold") == heldout_fold
    ).select("inchikey_full").unique()

    if expected_ids.is_empty():
        raise ValueError(f"No truth rows for held-out fold {heldout_fold}")

    actual_ids = predictions.select("inchikey_full")

    missing = expected_ids.join(
        actual_ids, on="inchikey_full", how="anti"
    )

    if not missing.is_empty():
        raise ValueError(f"Missing molecules: {missing.head(5).to_dicts()}")

    extra = actual_ids.join(
        expected_ids, on="inchikey_full", how="anti"
    )

    if not extra.is_empty():
        raise ValueError(f"Unexpected molecules: {extra.head(5).to_dicts()}")

    # Conversion preserves the correct ID → target → prediction mapping
    prediction_rows = []

    for target in _SCORED_DEFAULT:
        for molecule, value in predictions.select(
            "inchikey_full", target
        ).iter_rows():
            prediction_rows.append({
                "model_id": model_id, 
                "inchikey_full": molecule,
                "fold": heldout_fold,
                "target": target,
                "pred": value,
            })
     
    converted = pl.DataFrame(
        prediction_rows,
        schema={
            "model_id": pl.String,
            "inchikey_full": pl.String,
            "fold": pl.Int64,
            "target": pl.String,
            "pred": pl.Float64,
        }
    )

    keys = ["inchikey_full", "fold", "target"]
    expected_keys = truth.filter(
        pl.col("fold") == heldout_fold
    ).select("inchikey_full", "fold", "target")

    converted = converted.join(
        expected_keys, 
        on=keys,
        how="semi"
    )

    _validate_inputs(
        truth=truth,
        heldout_fold=heldout_fold,
        predictions=converted,
        model_id=model_id
    )

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or resume out-of-fold predictions using saved fold assignments."
    )
    parser.add_argument(
        "--data-path",
        type=Path, 
        required=True, 
        help="Molecule CSV with inchikey_full, inchikey_block, SMILES, and four scored targets; "
             "reserved validation/test molecules must already be excluded. "
    )
    parser.add_argument(
        "--folds-path",
        type=Path, 
        required=True,
        help="Saved CSV with inchikey_full and integer fold columns; "
             "assignments are reused unchanged.",
    )
    parser.add_argument(
        "--model-name",
        choices=[*REGISTRY, "chemprop"],
        required=True,
        help="Registered feature-based or chemprop model to train for each fold.",
    )
    parser.add_argument(
        "--model-id",
        type=str, 
        required=True,
        help="Unique label for this model configuration, stored in predictions.",
    )
    parser.add_argument(
        "--model-params",
        default="{}",
        help='Model constructor parameters as a JSON object, e.g. \'{"max_depth": 5}\'.',
    )
    parser.add_argument(
        "--input-names", 
        nargs="+",
        required=False,
        help="One or more feature representations used to train the model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for fold artifacts and combined outputs; "
             "use a separate directory for each model configuration.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=False,
        help="Directory for cached molecular features.",
    )
    parser.add_argument(
        "--chemprop-config",
        required=False,
        help=(
        "JSON object required for --model-name chemprop. "
        "Required keys: epochs, seed, target_columns, from_foundation "
        "(null, 'chemeleon', or a local checkpoint path). "
        "Optional: training_timeout and prediction_timeout in seconds."
    ),
    )
    parser.add_argument(
        "--group-column",
        type=str,
        required=False,
        help="Identifies groups that must be kept together during " 
             "internal-validation selection.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        required=False,
        help="Requested approximate fraction of non-heldout molecules used " 
             " for internal validation. It is approximate because groups remain intact.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        required=False,
        help="Seed for internal validation.",
    )

    args = parser.parse_args()

    try:
        model_params = json.loads(args.model_params)
    except json.JSONDecodeError as exc:
        parser.error(f"--model-params must be valid JSON: {exc}")

    if not isinstance(model_params, dict):
        parser.error("--model-params must be a JSON object")

    chemprop_config = None

    if args.chemprop_config is not None:
        try:
            chemprop_config = json.loads(args.chemprop_config)
        except json.JSONDecodeError as exc:
            parser.error(f"--chemprop-config must be valid JSON: {exc}")

        if not isinstance(chemprop_config, dict):
            parser.error("--chemprop-config must be a JSON object")
        
    if args.model_name == "chemprop":
        if args.chemprop_config is None:
            parser.error("--chemprop-config is required for Chemprop")
        if args.group_column is None:
            parser.error("--group-column is required for Chemprop")
        if args.validation_fraction is None:
            parser.error("--validation-fraction is required for Chemprop")
        if args.validation_seed is None:
            parser.error("--validation-seed is required for Chemprop")
    else:
        if args.input_names is None:
            parser.error("--input-names is required for feature-based models")
        unknown = [
            name for name in args.input_names
            if not is_known_input(name)
        ]
        if unknown:
            parser.error(f"--input-names: unknown input(s): {unknown}")
        if args.cache_dir is None:
            parser.error("--cache-dir is required for feature-based models")
        
    run_oof(
        data_path=args.data_path, 
        folds_path=args.folds_path,
        model_name=args.model_name,
        model_id=args.model_id,
        model_params=model_params,
        input_names=args.input_names,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        chemprop_config=chemprop_config,
        group_column=args.group_column,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )

    print(f"Saved OOF predictions and truth to {args.output_dir}")

if __name__ == "__main__":
    main()
