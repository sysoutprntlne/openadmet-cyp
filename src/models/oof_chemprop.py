"""Train and predict with Chemprop for individual OOF folds.

Create group-aware internal validation splits, validate Chemprop settings,
and convert held-out predictions to the shared OOF format.
"""

import math
import random
import subprocess
import sys
from pathlib import Path

import polars as pl

from models.oof import _SCORED_DEFAULT, _convert_chemprop_predictions
from models.training.chemprop import build_command


def _generate_chemprop_fold_predictions(
    train_df: pl.DataFrame, 
    val_df: pl.DataFrame, 
    heldout_df: pl.DataFrame, 
    truth: pl.DataFrame, 
    model_id: str, 
    heldout_fold: int, 
    work_dir: Path, 
    target_columns: list[str],
    epochs: int, 
    seed: int,
    from_foundation: str | None = None,
    training_timeout: int = 300,
    prediction_timeout: int = 300,
    accelerator: str = "cpu",
    devices: int = 1,
    num_workers: int = 0,
    batch_size: int = 4,
    warmup_epochs: int = 0,
) -> pl.DataFrame:
    """Train Chemprop on fitting rows, select the best checkpoint using
    internal validation, and predict the held-out OOF rows.

    Require disjoint connectivity blocks across the three populations and an
    empty work directory. Auxiliary targets may be included in training;
    return long-format OOF predictions for the four scored targets only.
    """

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

    truth_required = {"inchikey_full", "fold", "target", "truth"}
    truth_missing = truth_required - set(truth.columns)
    if truth_missing:
        raise ValueError(f"truth is missing required columns: {sorted(truth_missing)}")

    heldout_df_required = {"inchikey_full", "inchikey_block", "SMILES"}
    heldout_df_missing = heldout_df_required - set(heldout_df.columns)
    if heldout_df_missing:
        raise ValueError(f"heldout_df is missing required columns: {sorted(heldout_df_missing)}")

    for name, df in [
        ("train_df", train_df),
        ("val_df", val_df),
        ("heldout_df", heldout_df),
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
    heldout_ids = set(heldout_df["inchikey_block"].to_list())

    if train_ids & val_ids:
        raise ValueError("Training and validation molecules overlap")
    if train_ids & heldout_ids:
        raise ValueError("Training and heldout molecules overlap")
    if val_ids & heldout_ids:
        raise ValueError("Validation and heldout molecules overlap")

    if len(train_df) == 0:
        raise ValueError("train_df is empty")
    
    if len(val_df) == 0:
        raise ValueError("val_df is empty")
    
    if len(heldout_df) == 0:
        raise ValueError("heldout_df is empty")
    
    if len(truth) == 0:
        raise ValueError("truth table is empty")

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

    if not any(
        val_df[target].drop_nulls().len() > 0
        for target in target_columns
    ):
        raise ValueError("Validation data contains no observed target labels")

    columns = ["inchikey_full", "SMILES", *target_columns]

    training_data = pl.concat([
        train_df.select(columns).with_columns(
            pl.lit("train").alias("split")
        ),
        val_df.select(columns).with_columns(
            pl.lit("val").alias("split")
        ),
    ])

    scored_ids = (
        truth.filter(pl.col("fold") == heldout_fold)
        .select("inchikey_full")
        .unique()
    )

    prediction_data = (
        heldout_df.join(scored_ids, on="inchikey_full", how="semi")
        .select("inchikey_full", "SMILES")
    )

    work_dir.mkdir(parents=True, exist_ok=True)

    if any(work_dir.iterdir()):
        raise ValueError(
            f"work_dir must be empty for a new training attempt: {work_dir}"
        )

    training_path = work_dir / "training.csv"
    heldout_path = work_dir / "heldout.csv"
    output_dir = work_dir / "chemprop"
    predictions_path = work_dir / "native_predictions.csv"

    training_data.write_csv(training_path)
    prediction_data.write_csv(heldout_path)

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
        "--test-path", str(heldout_path),
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
    
    return _convert_chemprop_predictions(
        predictions=pl.read_csv(predictions_path),
        truth=truth,
        model_id=model_id,
        heldout_fold=heldout_fold,
    )

def _split_internal_validation(
    training_pool: pl.DataFrame, # pinned training with OOF heldout removed
    *, 
    group_column: str,
    validation_fraction: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split non-heldout molecules into fitting and validation tables."""

    if training_pool.is_empty():
        raise ValueError("training_pool is empty")

    required = {"inchikey_full", group_column, *_SCORED_DEFAULT}
    missing = required - set(training_pool.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(validation_fraction)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("validation_fraction must be finite and between 0 and 1")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    for column in {"inchikey_full", group_column}:
        values = training_pool[column]

        if values.dtype != pl.String:
            raise ValueError(f"{column} must contain strings")
        if values.null_count() > 0:
            raise ValueError(f"{column} contains null values")
        if (values.str.strip_chars() == "").any():
            raise ValueError(f"{column} contains blank values")

    if training_pool["inchikey_full"].n_unique() != training_pool.height:
        raise ValueError("training_pool contains duplicate molecule IDs")

    group_sizes = (
        training_pool
        .group_by(group_column)
        .agg(pl.len().alias("n_molecules"))
        .sort(group_column)
    )

    if group_sizes.height < 2:
        raise ValueError("At least two groups are required")

    groups = list(group_sizes.iter_rows(named=True))

    rng = random.Random(seed)
    rng.shuffle(groups)

    target_count = validation_fraction * training_pool.height

    running_count = 0
    best_count = 0
    best_distance = float("inf")

    for count, group in enumerate(groups[:-1], start =1):
        running_count += group["n_molecules"]
        distance = abs(running_count - target_count)

        if distance < best_distance:
            best_distance = distance
            best_count = count

    validation_groups = [
        group[group_column]
        for group in groups[:best_count]
    ]

    is_validation = pl.col(group_column).is_in(validation_groups)

    for target in _SCORED_DEFAULT:
        values = training_pool[target]

        if values.dtype != pl.Null and not values.dtype.is_numeric():
            raise ValueError(f"{target} must be numeric")

        if values.cast(pl.Float64).is_infinite().any():
            raise ValueError(f"{target} contains infinite values")

    training_pool = training_pool.with_columns([
        pl.col(target).cast(pl.Float64).fill_nan(None)
        for target in _SCORED_DEFAULT
    ])


    val_df = training_pool.filter(is_validation)
    train_df = training_pool.filter(~is_validation)

    for target in _SCORED_DEFAULT:
        observed = train_df[target].drop_nulls()
        if observed.is_empty():
            raise ValueError(f"No training labels for scored target {target}")

    if not any(
        val_df[target].drop_nulls().len() > 0
        for target in _SCORED_DEFAULT
    ):
        raise ValueError("Validation data contains no observed target labels")
    
    return train_df, val_df


def _resolve_chemprop_config(config: dict) -> dict:
    """Validate Chemprop settings, fill timeout defaults, and normalize foundation paths."""

    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    if any(not isinstance(key, str) for key in config):
        raise ValueError("config keys must be strings")
    
    required = {
        "epochs",
        "seed",
        "from_foundation",
        "target_columns",
    }

    missing = required - set(config)
    if missing:
        raise ValueError(f"config must include: {sorted(missing)}")

    optional_defaults = {
        "training_timeout": 300,
        "prediction_timeout": 300,
        "accelerator": "cpu",
        "devices": 1,
        "num_workers": 0,
        "batch_size": 4,
        "warmup_epochs": 0,
    }

    allowed = required | set(optional_defaults)
    unknown = set(config) - allowed

    if unknown:
        raise ValueError(f"config includes unexpected keys: {sorted(unknown)}")

    resolved = {**optional_defaults, **config}
    accelerator = resolved["accelerator"]
    if (
        not isinstance(accelerator, str)
        or accelerator not in {"cpu", "gpu", "cuda", "mps", "auto"}
    ):
        raise ValueError("accelerator must be cpu, gpu, cuda, mps, or auto")

    for name, minimum in [
        ("devices", 1),
        ("num_workers", 0),
        ("batch_size", 1),
        ("warmup_epochs", 0),
    ]:
        value = resolved[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")

    epochs = resolved["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be positive integers")

    if resolved["warmup_epochs"] >= epochs:
        raise ValueError("warmup_epochs must be less than epochs")
    
    if not isinstance(resolved["seed"], int) or isinstance(resolved["seed"], bool):
        raise ValueError("Seed must be integer.") 
    
    target_columns = resolved["target_columns"]

    if not isinstance(target_columns, list) or not target_columns:
        raise ValueError("target_columns must be a nonempty list")

    if any(
        not isinstance(target, str) or not target.strip()
        for target in target_columns
    ):
        raise ValueError("target_columns must contain nonblank strings")

    if len(target_columns) != len(set(target_columns)):
        raise ValueError("target_columns contains duplicate names")
    
    reserved = {"inchikey_full", "inchikey_block", "SMILES", "split", "fold", "model_id"}

    conflicts = set(target_columns) & reserved
    if conflicts:
        raise ValueError(f"Target names conflict with metadata columns: {sorted(conflicts)}")

    missing = set(_SCORED_DEFAULT) - set(target_columns)
    if missing:
        raise ValueError(f"Missing scored targets: {sorted(missing)}")

    for name in ("training_timeout", "prediction_timeout"):
        value = resolved[name]

        if (
            isinstance(value, bool) 
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")

    foundation = resolved["from_foundation"]

    if foundation is not None:
        if not isinstance(foundation, str) or not foundation.strip():
            raise ValueError(
                "from_foundation must be None, 'chemeleon', or a checkpoint path"
            )

        if foundation.lower() == "chemeleon":
            resolved["from_foundation"] = "chemeleon"
        else:
            checkpoint = Path(foundation).expanduser()

            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Foundation checkpoint does not exist: {checkpoint}"
                )

            resolved["from_foundation"] = str(checkpoint.resolve())

    resolved["target_columns"] = list(target_columns)
    
    return resolved

  