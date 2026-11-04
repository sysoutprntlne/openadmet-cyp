"""Assemble one wide multi-task table: challenge labels + external auxiliary heads.

The design decision this file encodes: external measurements are never pooled
into the scored pIC50 columns. Each source x isoform x readout becomes its
own column -- its own head at training time -- so no cross-assay calibration
is needed. A shared D-MPNN encoder sees every molecule and every label; each
head keeps its own scale. Chemprop masks missing targets in the loss
automatically, so the sparse block structure below costs nothing.

This replaces the earlier pretrain-then-finetune framing. Rather than fitting
an encoder on pooled external affinity and then transferring it, we train one
model jointly on all heads at once, which avoids both the pooling
(cross-assay) problem and catastrophic forgetting during fine-tuning.

Two keys, used for different jobs. Sources are *joined* on the InChIKey
connectivity block, so a salt in one source matches the free base in another.
But rows *within* the challenge deck are collapsed on the full InChIKey,
because stereoisomers share a connectivity block while being different
molecules with different labels -- quinidine and quinine differ by 2.56 log
units on CYP2D6. Collapsing those on connectivity would average real signal
away.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from data_tools.overlap import add_keys, add_stereo_keys, collapse_duplicates, overlap_report

_KEY = "inchikey_block"
_STEREO_KEY = "inchikey_full"

_CHALLENGE_TARGETS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]


def _auxiliary_columns(df: pl.DataFrame, exclude: list[str]) -> list[str]:
    """Return numeric label columns of an auxiliary source, ignoring identifiers."""
    skip = set(exclude) | {"SMILES", _KEY, "PUBCHEM_CID", "source", "affinity_type"}
    return [c for c in df.columns if c not in skip and df[c].dtype.is_numeric()]


def build(
    challenge_path: Path,
    auxiliary_paths: dict[str, Path],
    challenge_targets: list[str],
    test_path: Path | None,
    drop_test_overlap: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """Join the challenge table with every auxiliary source on connectivity block.

    Returns (union_df, head_columns) where head_columns lists every label
    column in training order: challenge heads first, then auxiliary heads.
    """
    print(f"Loading challenge labels <- {challenge_path}", file=sys.stderr)
    # infer_schema_length=None throughout: sparse label columns that are null
    # for the first 100 rows would otherwise be typed String, silently breaking
    # every numeric aggregation downstream.
    challenge = add_stereo_keys(add_keys(pl.read_csv(challenge_path, infer_schema_length=None)))
    present_targets = [t for t in challenge_targets if t in challenge.columns]
    missing = [t for t in challenge_targets if t not in challenge.columns]
    if missing:
        print(f"  warning: challenge targets not found and skipped: {missing}", file=sys.stderr)
    n_before = len(challenge)
    # Collapse on full InChIKey, not connectivity block: stereoisomers are
    # distinct molecules with distinct labels and must survive as separate rows.
    challenge = collapse_duplicates(
        challenge.select([_STEREO_KEY, _KEY, "SMILES"] + present_targets),
        present_targets,
        key=_STEREO_KEY,
    )
    print(
        f"  {n_before} rows -> {len(challenge)} stereo-distinct molecules "
        f"({challenge[_KEY].n_unique()} connectivity skeletons), {len(present_targets)} scored heads",
        file=sys.stderr,
    )

    union = challenge
    head_columns = list(present_targets)

    for name, path in auxiliary_paths.items():
        print(f"Loading auxiliary source {name!r} <- {path}", file=sys.stderr)
        aux = add_keys(pl.read_csv(path, infer_schema_length=None))
        aux_heads = _auxiliary_columns(aux, exclude=present_targets)
        if not aux_heads:
            print(f"  warning: no numeric label columns found in {path}, skipping", file=sys.stderr)
            continue
        aux = collapse_duplicates(aux.select([_KEY, "SMILES"] + aux_heads), aux_heads)
        shared = overlap_report(challenge, aux, "challenge", name)
        print(f"  {len(aux)} unique skeletons, {len(aux_heads)} auxiliary heads", file=sys.stderr)

        union = union.join(aux.select([_KEY] + aux_heads), on=_KEY, how="full", coalesce=True)
        # Molecules present only in the auxiliary source arrive with a null SMILES
        # from the challenge side; backfill from the auxiliary table.
        aux_smiles = aux.select([_KEY, pl.col("SMILES").alias("_aux_smiles")])
        union = (
            union.join(aux_smiles, on=_KEY, how="left", coalesce=True)
            .with_columns(pl.coalesce([pl.col("SMILES"), pl.col("_aux_smiles")]).alias("SMILES"))
            .drop("_aux_smiles")
        )
        head_columns += aux_heads
        print(f"  union now {len(union)} skeletons, {shared} of them shared with the challenge deck", file=sys.stderr)

    if test_path is not None and test_path.exists():
        print(f"\nLeakage check against blind test set <- {test_path}", file=sys.stderr)
        test = add_keys(pl.read_csv(test_path, infer_schema_length=None))
        overlap_report(union, test, "union", "test")
        if drop_test_overlap:
            test_keys = set(test[_KEY].to_list())
            leaked = union.filter(pl.col(_KEY).is_in(test_keys))
            if len(leaked):
                print(
                    f"  dropping {len(leaked)} union row(s) whose skeleton appears in the blind set: "
                    f"{leaked[_KEY].to_list()}",
                    file=sys.stderr,
                )
                union = union.filter(~pl.col(_KEY).is_in(test_keys))

    union = union.filter(pl.col("SMILES").is_not_null())
    return union.select(["SMILES"] + head_columns + [_KEY]), head_columns


def summarize(union: pl.DataFrame, head_columns: list[str]) -> None:
    """Print per-head label counts and the total label budget."""
    print(f"\n=== union: {len(union)} molecules, {len(head_columns)} heads ===")
    width = max(len(h) for h in head_columns)
    total = 0
    for head in head_columns:
        n = len(union) - union[head].null_count()
        total += n
        print(f"  {head:{width}}  {n:>7} labels")
    print(f"  {'TOTAL':{width}}  {total:>7} labels")


def main() -> None:
    """Entry point for the build-union CLI."""
    parser = argparse.ArgumentParser(
        description="Build one wide multi-task CSV from the challenge labels plus external "
        "auxiliary sources, joined on InChIKey connectivity block. One column per head.",
    )
    parser.add_argument("--challenge-path", type=Path, default=Path("data/train.csv"), help="Challenge train CSV")
    parser.add_argument(
        "--auxiliary",
        nargs="*",
        default=["pubchem=data/pubchem_aid1851.csv"],
        metavar="NAME=PATH",
        help="Auxiliary sources as NAME=PATH (default: pubchem=data/pubchem_aid1851.csv)",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/test.csv"),
        help="Blind test CSV, used only to report leakage (default: data/test.csv)",
    )
    parser.add_argument(
        "--challenge-targets",
        nargs="+",
        default=_CHALLENGE_TARGETS,
        help="Scored challenge label columns to carry through",
    )
    parser.add_argument("--output-path", type=Path, default=Path("data/union_train.csv"), help="Output CSV")
    parser.add_argument(
        "--keep-test-overlap",
        action="store_true",
        help="Keep union rows whose skeleton appears in the blind test set (default: drop them)",
    )
    args = parser.parse_args()

    auxiliary_paths: dict[str, Path] = {}
    for spec in args.auxiliary:
        if "=" not in spec:
            print(f"--auxiliary entries must be NAME=PATH, got {spec!r}", file=sys.stderr)
            sys.exit(1)
        name, _, path_text = spec.partition("=")
        path = Path(path_text)
        if not path.exists():
            print(f"Missing auxiliary file: {path} (run `data fetch pubchem_aid1851` first?)", file=sys.stderr)
            sys.exit(1)
        auxiliary_paths[name] = path

    union, head_columns = build(
        args.challenge_path,
        auxiliary_paths,
        args.challenge_targets,
        args.test_path,
        drop_test_overlap=not args.keep_test_overlap,
    )
    summarize(union, head_columns)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    union.write_csv(args.output_path)
    print(f"\nWrote {args.output_path}", file=sys.stderr)
    print("Train all heads jointly with:", file=sys.stderr)
    print(
        f"  train-chemprop --data-path {args.output_path} --target-columns {' '.join(head_columns)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
