"""Data handling shared by every training backend.

Datasets are chosen with `--datasets NAME[:ISO,ISO]` and merged at run time into
one wide table with a head per source x isoform. The ADMET data is always part of
the final stage: it supplies the four scored heads and the pinned split.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_tools.merge_datasets import merge
from data_tools.overlap import add_keys


def slug(specs: list[str]) -> str:
    """Directory-safe name for a list of dataset specs."""
    return "+".join(specs).replace(":", "-").replace(",", "_") or "admet"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags every backend shares."""
    parser.add_argument(
        "--datasets",
        nargs="+",
        action="append",
        default=None,
        metavar="NAME[:ISO,ISO]",
        help="One dataset config; repeat --datasets for several. Only the datasets you list are trained on: 'admet' "
        "is the ADMET data, so --datasets admet pubchem:CYP1A2 trains on both and --datasets pubchem:CYP1A2 on the "
        "auxiliary data alone (the ADMET heads are then scored with the same-isoform auxiliary head). Heads are "
        "named {isoform}-{source}, e.g. bindingdb:CYP2D6,CYP1A2 tdi. Every config is trained once per seed.",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List selectable datasets and exit")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where the source CSVs live (default: data)")
    parser.add_argument("--split-strategy", default="scaffold", choices=["scaffold", "butina", "random"])
    parser.add_argument("--split-seed", type=int, default=0, help="Seed for the merged table's split (default: 0)")
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=None,
        help="Heads to train on (default: every head of the selected datasets). Missing labels are masked in the loss.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for outputs (default: results/<model>/<config>; with several runs, <output-dir>/<config>/seed<N>)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, or the first seed with --n-seeds (default: 42)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Train every config once per seed listed")
    parser.add_argument("--n-seeds", type=int, default=None, help="Train every config with seeds --seed .. --seed+N-1")


def merge_stage(specs: dict[str, list[str] | None], args: argparse.Namespace):
    """Merge the challenge with the given sources on the pinned split.

    Returns (table, heads, scored, tasks) exactly as data_tools.merge_datasets.merge.
    """
    return merge(
        {"challenge": None, **specs},  # always loaded for the pinned split and scoring; its labels train only if listed
        data_dir=args.data_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )


def prepare_joint_split(
    specs: dict[str, list[str] | None], args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    """Train/val frames plus (target columns, {scored head: head whose prediction scores it}) for Python-API backends.

    Train holds every pinned-train molecule with at least one selected label, so
    auxiliary-only molecules still train their head. Val is the pinned val split.
    """
    table, heads, scored, tasks = merge_stage(specs, args)
    if not args.use_admet:
        target_columns = args.target_columns or [h for h in heads if h not in scored]
        if any(h in scored for h in target_columns):
            parser.error("--target-columns lists ADMET heads but the config does not include admet")
        if not target_columns:
            parser.error("nothing to train on: list admet or an auxiliary dataset in --datasets")
    else:
        target_columns = args.target_columns or heads
    unknown = [c for c in target_columns if c not in heads]
    if unknown:
        parser.error(f"--target-columns not among heads {heads}: {unknown}")
    if not args.use_admet:
        # Score each ADMET head with the first auxiliary head of the same isoform (CYP1A2-Direct <- CYP1A2-PubChem).
        eval_map = {}
        for head in scored:
            prefix = head.split("-")[0] + "-"
            matches = [c for c in target_columns if c.startswith(prefix)]
            # percent-inhibition heads are on a different scale from pIC50; prefer any potency head
            match = next((c for c in matches if "pctinhib" not in c), matches[0] if matches else None)
            if match is not None:
                eval_map[head] = match
                print(f"  scoring {head} with {match}", file=sys.stderr)
    else:
        eval_map = {h: h for h in target_columns if h in scored}
    if not eval_map:
        parser.error(f"nothing to score: need a scored head ({scored}) or an auxiliary head of the same isoform")
    if any(tasks[h] == "classification" for h in target_columns):
        print("  binary heads present: training as regression, 0/1 labels as numbers", file=sys.stderr)

    df = pd.DataFrame(add_keys(table).to_dict(as_series=False))  # avoids the pyarrow dependency of .to_pandas()
    has_label = df[target_columns].notna().any(axis=1) & df["SMILES"].notna()
    train_df = df.loc[(df["split"] == "train") & has_label].copy()
    val_df = df.loc[df["split"] == "val"].copy()

    if train_df.empty:
        raise ValueError("No training examples")
    if not val_df[list(eval_map)].notna().any().any():
        raise ValueError(f"No validation examples for {list(eval_map)}")

    # Guard against structural leakage between train and validation.
    overlap = set(train_df["inchikey_block"]) & set(val_df["inchikey_block"])
    if overlap:
        raise ValueError(f"Train/validation overlap: {len(overlap)} connectivity blocks")
    return train_df, val_df, target_columns, eval_map


def write_val_outputs(
    output_dir: Path,
    run_id: str,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    pred_by_head: dict[str, np.ndarray],
    eval_map: dict[str, str],
    run_info: dict[str, object],
) -> None:
    """Score each scored head on val; write predictions.csv (T4 long format) and metrics.csv."""
    long_frames, metric_rows = [], []
    for head, source in eval_map.items():
        labelled = val_df[head].notna().to_numpy()
        actual = val_df.loc[labelled, head].to_numpy(dtype=float)
        pred = pred_by_head[source][labelled]
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        model_id = f"{run_id}_{head}"
        long_frames.append(pd.DataFrame({
            "model_id": model_id,
            "inchikey_block": val_df.loc[labelled, "inchikey_block"].to_numpy(),
            "fold": "val",
            "target": head,
            "pred": pred,
        }))
        metric_rows.append({
            "model_id": model_id,
            "target": head,
            "pred_head": source,
            **run_info,
            "n_train": int(train_df[source].notna().sum()),
            "n_val": int(labelled.sum()),
            "val_rmse": rmse,
        })
        print(f"Validation RMSE {head}: {rmse:.4f}")

    prediction_path = output_dir / "predictions.csv"
    pd.concat(long_frames).to_csv(prediction_path, index=False)
    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f"Macro validation RMSE: {np.mean([r['val_rmse'] for r in metric_rows]):.4f}")
    print(f"Predictions saved to: {prediction_path}")
    print(f"Metrics saved to: {metrics_path}")

