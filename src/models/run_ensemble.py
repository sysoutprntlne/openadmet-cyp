"""Run seed-averaged OOF and validation predictions, filter and cluster
candidates, fit a stacker, and save predictions, metrics, and artifacts.

Requires input data with pinned train/val/test splits.
"""
import argparse
import json
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from data_tools.prepare_ensemble import (
    SCORED_TARGETS,
    make_oof_folds,
    prepare_ensemble_data,
)
from models.ensemble import fit_filtered_stack
from models.evaluate import _compute_metrics
from models.oof import _load_oof_data, run_oof_seeds
from models.stack import _seed_averaging as average_oof_seeds
from models.validation import (
    _seed_averaging as average_validation_seeds,
    run_validation_seeds,
)


def _validation_metrics(predictions, truth, ensemble_predictions) -> pl.DataFrame:
    """Compare candidates and ensemble on identical observed rows per target."""
    
    rows = []
    keys = ["inchikey_full", "target"]

    for target in sorted(truth["target"].unique().to_list()):
        target_truth = truth.filter(pl.col("target") == target)
        candidates = predictions.filter(pl.col("target") == target)
        outputs = [
            ("individual", model_id, candidates.filter(pl.col("model_id") == model_id))
            for model_id in sorted(candidates["model_id"].unique().to_list())
        ]
        outputs.append((
            "ensemble", "ensemble",
            ensemble_predictions.filter(pl.col("target") == target),
        ))
        for kind, model_id, table in outputs:
            aligned = target_truth.join(
                table.select(*keys, "pred"), on=keys, how="left", validate="1:1",
            )
            if (
                table.height != target_truth.height
                or aligned["pred"].null_count()
                or not aligned["pred"].is_finite().all()
            ):
                raise ValueError(f"Validation rows do not match for {target}/{model_id}")
            rows.append({
                "evaluation": "development",
                "kind": kind,
                "model_id": model_id,
                "target": target,
                "n_rows": aligned.height,
                **_compute_metrics(aligned["truth"], aligned["pred"]),
            })
    return pl.DataFrame(rows)


def run(config: dict) -> None:
    """Run the ensemble workflow from configuration and save outputs."""

    models = config["models"]
    if not models:
        raise ValueError("At least one model configuration is required")

    configuration_ids = [model["configuration_id"] for model in models]
    if any(
        not isinstance(value, str) or not value.strip()
        for value in configuration_ids
    ):
        raise ValueError("Configuration IDs must be nonblank strings")

    if any(
        value in {".", ".."} or "/" in value or "\\" in value
        for value in configuration_ids
    ):
        raise ValueError("Configuration IDs must be simple directory names")

    if len(configuration_ids) != len(set(configuration_ids)):
        raise ValueError("Duplicate configuration IDs")

    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("Output path must be a directory")
        if any(output_dir.iterdir()):
            raise ValueError("Output directory must be empty")

    if config["stacker"] not in {"nnls", "ridgecv", "elasticnetcv"}:
        raise ValueError(f"Unknown stacking method: {config['stacker']}")
        
    data = pl.read_csv(config["data_path"], infer_schema_length=None)
    prepared = prepare_ensemble_data(data)

    if "oof_group" in prepared.columns:
        raise ValueError(
            "Input data must not contain oof_group; "
            "groups are supplied by the fold assignments."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if config["folds"]["path"] is None:
        fold_assignments = make_oof_folds(
            prepared,
            strategy=config["folds"]["strategy"],
            n_folds=config["folds"]["n_folds"],
            seed=config["folds"]["seed"],
            butina_cutoff=config["folds"].get("butina_cutoff", 0.4),
        )
    else:
        fold_assignments = pl.read_csv(
            config["folds"]["path"], infer_schema_length=None
        )

    required = {"inchikey_full", "fold", "oof_group"}
    if required - set(fold_assignments.columns):
        raise ValueError("Fold mapping requires inchikey_full, fold, and oof_group")

    groups = fold_assignments["oof_group"]
    if groups.dtype != pl.String or groups.null_count():
        raise ValueError("oof_group must contain non-null strings")
    if (groups.str.strip_chars() == "").any():
        raise ValueError("oof_group cannot be blank")

    group_folds = fold_assignments.group_by("oof_group").agg(
        pl.col("fold").n_unique().alias("n_folds")
    )
    if (group_folds["n_folds"] > 1).any():
        raise ValueError("An OOF group appears in multiple folds")

    block_groups = (
        prepared.filter(pl.col("split") == "train")
        .select("inchikey_full", "inchikey_block")
        .join(
            fold_assignments.select("inchikey_full", "oof_group"),
            on="inchikey_full",
            how="left",
            validate="1:1",
        )
    )

    if block_groups["oof_group"].null_count():
        raise ValueError("Missing OOF groups for training molecules")

    group_counts = block_groups.group_by("inchikey_block").agg(
        pl.col("oof_group").n_unique().alias("n_groups")
    )
    if (group_counts["n_groups"] > 1).any():
        raise ValueError("A connectivity block appears in multiple OOF groups")

    prepared = prepared.join(
        fold_assignments.select("inchikey_full", "oof_group"),
        on="inchikey_full",
        how="left",
        validate="1:1",
    )

    train = prepared.filter(pl.col("split") == "train")

    train_path = output_dir / "train.csv"
    folds_path = output_dir / "folds.csv"
    train.write_csv(train_path)
    fold_assignments.write_csv(folds_path)

    # Validation
    _load_oof_data(train_path, folds_path)

    with (output_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)

    prepared.write_csv(output_dir / "prepared.csv")
    
    oof_tables = []
    validation_tables = []
    first_truth_oof = None
    first_truth_val = None
    for model_config in config["models"]:
        configuration_id = model_config["configuration_id"]

        oof_by_seed, oof_truth = run_oof_seeds(
            data_path=train_path,
            folds_path=folds_path,
            model_name=model_config["model_name"],
            model_id=configuration_id,
            model_params=model_config.get("model_params", {}),
            input_names=model_config.get("input_names"),
            output_dir=output_dir / configuration_id / "oof",
            cache_dir=output_dir / "features",
            chemprop_config=model_config.get("chemprop_config"),
            group_column="oof_group",
            validation_fraction=model_config.get("validation_fraction", 0.1),
            validation_seed=model_config.get("validation_seed", 42),
            seeds=config["seeds"],
        )
        model_dir = output_dir / configuration_id
        model_dir.mkdir(parents=True, exist_ok=True)
        for seed, predictions in oof_by_seed.items():
            seed_dir = model_dir / "oof" / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            predictions.write_csv(seed_dir / "oof_predictions.csv")
        oof_truth.write_csv(model_dir / "oof_truth.csv")

        validation_by_seed, validation_truth = run_validation_seeds(
            df=prepared,
            model_name=model_config["model_name"],
            model_id=configuration_id,
            model_params=model_config.get("model_params", {}),
            input_names=model_config.get("input_names"),
            output_dir=output_dir / configuration_id / "validation",
            cache_dir=output_dir / "features",
            chemprop_config=model_config.get("chemprop_config"),
            group_column="oof_group",
            validation_fraction=model_config.get("validation_fraction", 0.1),
            validation_seed=model_config.get("validation_seed", 42),
            seeds=config["seeds"],
            targets=SCORED_TARGETS,
        )
        for seed, predictions in validation_by_seed.items():
            seed_dir = model_dir / "validation" / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            predictions.write_csv(seed_dir / "predictions.csv")
        validation_truth.write_csv(model_dir / "validation_truth.csv")
        oof_truth = oof_truth.sort(["inchikey_full", "fold", "target"])
        if first_truth_oof is None:
            first_truth_oof = oof_truth
        else:
            assert_frame_equal(first_truth_oof, oof_truth)

        validation_truth = validation_truth.sort(["inchikey_full", "target"])
        if first_truth_val is None:
            first_truth_val = validation_truth
        else:
            assert_frame_equal(first_truth_val, validation_truth)

        averaged_oof = average_oof_seeds(
            oof_by_seed, first_truth_oof, model_id=configuration_id,
        )
        averaged_validation = average_validation_seeds(
            validation_by_seed, first_truth_val, model_id=configuration_id
        )
        averaged_oof.write_csv(model_dir / "averaged_oof.csv")
        averaged_validation.write_csv(model_dir / "averaged_validation.csv")

        oof_tables.append(averaged_oof)
        validation_tables.append(averaged_validation)

    all_oof_predictions = pl.concat(oof_tables)
    all_validation_predictions = pl.concat(validation_tables)

    all_oof_predictions.write_csv(output_dir / "oof_predictions.csv")
    first_truth_oof.write_csv(output_dir / "oof_truth.csv")

    all_validation_predictions.write_csv(output_dir / "validation_predictions.csv")
    first_truth_val.write_csv(output_dir / "validation_truth.csv")

    result = fit_filtered_stack(
        validation_predictions=all_validation_predictions,
        validation_truth=first_truth_val,
        oof_predictions=all_oof_predictions,
        oof_truth=first_truth_oof,
        method=config["stacker"],
        on_filter_decisions=lambda decisions: decisions.write_csv(
            output_dir / "filter_decisions.csv"
        ),
    )
    result["cluster_membership"].write_csv(output_dir / "cluster_membership.csv")
    result["oof_predictions"].write_csv(output_dir / "clustered_oof_predictions.csv")
    result["validation_predictions"].write_csv(output_dir / "clustered_validation_predictions.csv")
    result["ensemble_validation_predictions"].write_csv(output_dir / "ensemble_validation_predictions.csv")
    # Stackers already convert fitted NumPy coefficients/parameters to Python floats.
    with (output_dir / "stack.json").open("w") as file:
        json.dump(result["stack"], file, indent=2, allow_nan=False)

    metrics = _validation_metrics(
        all_validation_predictions, first_truth_val,
        result["ensemble_validation_predictions"],
    )
    metrics.write_csv(output_dir / "validation_metrics.csv")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ensemble pipeline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open() as file:
        config = json.load(file)

    run(config)

if __name__ == "__main__":
    main()
