"""CLI for inspecting and filling the molecular-representation feature cache."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from data_tools.inputs import INPUT_REGISTRY, featurize, is_known_input
from data_tools.load import load_data, load_test, load_train
from representations.embedding_metrics import evaluate_embedding, evaluate_shift, fingerprint_shift
from representations.foundation_alias import ALIAS_FILE, set_alias

console = Console()


def _molecules(data_dir: Path) -> pl.DataFrame:
    """Unique SMILES of the challenge train and blinded test sets: what evaluate-models and generate-results featurize."""
    frames = [load_train(data_dir / "train.csv").select("SMILES"), load_test(data_dir / "test.csv").select("SMILES")]
    return pl.concat(frames).unique(subset=["SMILES"], keep="first", maintain_order=True)


def _size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError


def _check(cache_dir: Path, smiles: set[str] | None) -> None:
    """Print one row per cached representation, plus registered ones that have no cache yet."""
    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", pad_edge=False)
    table.add_column("Representation", style="bold", overflow="fold")
    table.add_column("Molecules", justify="right")
    table.add_column("Dim", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Updated", no_wrap=True)
    if smiles is not None:
        table.add_column("Covers", justify="right")

    cached: set[str] = set()
    for npy in sorted(cache_dir.glob("*.npy")) if cache_dir.exists() else []:
        idx = npy.with_suffix(".idx")
        if not idx.exists():
            table.add_row(escape(npy.stem), "[red]no .idx[/]", "", _size(npy.stat().st_size), "")
            continue
        cached.add(npy.stem)
        indexed = idx.read_text().splitlines()
        shape = np.load(npy, mmap_mode="r").shape
        stale = "[red]mismatch[/]" if shape[0] != len(indexed) else str(shape[0])
        row = [
            escape(npy.stem),
            stale,
            str(shape[1]) if len(shape) > 1 else "?",
            _size(npy.stat().st_size + idx.stat().st_size),
            time.strftime("%Y-%m-%d %H:%M", time.localtime(npy.stat().st_mtime)),
        ]
        if smiles is not None:
            have = len(smiles.intersection(indexed))
            colour = "green" if have == len(smiles) else "yellow" if have else "red"
            row.append(f"[{colour}]{have}/{len(smiles)}[/]")
        table.add_row(*row)

    for name in INPUT_REGISTRY:
        if name not in cached:
            table.add_row(f"[dim]{name}[/]", "[dim]not cached[/]", "", "", "", *([""] if smiles is not None else []))
    console.print(f"[dim]cache dir:[/] {cache_dir}  [dim]Covers = cached molecules out of {len(smiles)} train+test[/]" if smiles is not None else f"[dim]cache dir:[/] {cache_dir}")
    console.print(table)


_TARGET_SUFFIX = "_pIC50_direct_inhibition"
_BASELINE = "morgan"
_METRIC_LEGEND = (
    "Probe/kNN R2: val R2 of ridge / 5-NN on the embedding (higher better). "
    "Recall: overlap of embedding and Morgan-Tanimoto 10-NN (higher better). "
    "EffRank: effective rank, low means collapse. "
    "Rough: 1-NN label difference over random-pair difference (lower smoother, <1 is locality). "
    "NN ratio: median held-out->train NN distance over train leave-one-out (1.0 = looks like train, "
    "below 1 = closer to train than train is to itself). "
    "Out: share of the held-out set beyond the train 95th-percentile NN distance (0.05 = like train). "
    "AUC: train-vs-held-out classifier AUROC (0.5 = indistinguishable). "
    "MMD p: permutation p-value, small = distributions differ. "
    "Err rho: Spearman of val NN distance vs probe error (positive = distance predicts failure)."
)


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a).all(axis=1)
    return mask


def _fmt(value: float, digits: int = 3) -> str:
    return "[dim]n/a[/]" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _evaluate(args: argparse.Namespace) -> None:
    """Score embeddings on a train/val split: embedding quality plus train-to-val distance."""
    names = list(INPUT_REGISTRY) if args.evaluate == ["all"] else args.evaluate
    unknown = [n for n in names if not is_known_input(n)]
    if unknown:
        print(f"Unknown representation(s): {unknown}. Available: {list(INPUT_REGISTRY)} or chemprop[/path]", file=sys.stderr)
        sys.exit(1)

    train_df, val_df = load_data(
        split_type=args.split,
        val_fraction=args.val_split,
        seed=args.seed,
        butina_cutoff=args.butina_cutoff,
        data_dir=args.data_dir,
    )
    targets = [c for c in train_df.columns if c.endswith(_TARGET_SUFFIX)]
    if args.targets:
        missing = [t for t in args.targets if t not in targets]
        if missing:
            print(f"Unknown target(s): {missing}. Available: {targets}", file=sys.stderr)
            sys.exit(1)
        targets = args.targets
    console.print(
        f"[dim]split[/]={args.split}  [dim]seed[/]={args.seed}  "
        f"[dim]{len(train_df)} train / {len(val_df)} val molecules[/]  [dim]targets[/]={len(targets)}"
    )

    fp_train_all = featurize(train_df, [_BASELINE], args.cache_dir)
    fp_val_all = featurize(val_df, [_BASELINE], args.cache_dir)
    y_train_all = {t: train_df[t].cast(pl.Float64).to_numpy() for t in targets}
    y_val_all = {t: val_df[t].cast(pl.Float64).to_numpy() for t in targets}

    if args.test and not (args.data_dir / "test.csv").exists():
        print(f"{args.data_dir / 'test.csv'} not found, skipping the test distances (use --no-test to silence)", file=sys.stderr)
        args.test = False
    if args.test:
        # the model that is submitted trains on all of train.csv, so the test set is measured against all of it
        full_df = pl.concat([train_df, val_df])
        test_df = load_test(args.data_dir / "test.csv")
        fp_full_all = featurize(full_df, [_BASELINE], args.cache_dir)
        fp_test_all = featurize(test_df, [_BASELINE], args.cache_dir)
        console.print(f"[dim]test: {len(test_df)} molecules against all {len(full_df)} train.csv molecules[/]")

    rows: list[tuple[str, dict[str, float]]] = []
    test_rows: list[tuple[str, dict[str, float]]] = []
    baseline_shown = False
    for name in names:
        try:
            X_train_all = featurize(train_df, [name], args.cache_dir)
            X_val_all = featurize(val_df, [name], args.cache_dir)
            tr = _finite_rows(X_train_all, fp_train_all)
            va = _finite_rows(X_val_all, fp_val_all)
            if tr.sum() < 20 or va.sum() < 20:
                raise ValueError(f"only {tr.sum()} train / {va.sum()} val molecules have finite features")
            fp_train = fp_train_all[tr].astype(np.uint8)
            fp_val = fp_val_all[va].astype(np.uint8)
            if not baseline_shown:
                base, _ = fingerprint_shift(fp_train, fp_val)
                rows.append((f"{_BASELINE} Tanimoto (baseline)", {"nn_ratio": base["nn_ratio"], "frac_out": base["frac_out"]}))
                baseline_shown = True
            metrics = evaluate_embedding(
                X_train_all[tr],
                X_val_all[va],
                {t: y[tr] for t, y in y_train_all.items()},
                {t: y[va] for t, y in y_val_all.items()},
                fp_train,
                fp_val,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            if args.test:
                X_full_all = featurize(full_df, [name], args.cache_dir)
                X_test_all = featurize(test_df, [name], args.cache_dir)
                full = _finite_rows(X_full_all, fp_full_all)
                tst = _finite_rows(X_test_all, fp_test_all)
                if full.sum() < 20 or tst.sum() < 20:
                    raise ValueError(f"only {full.sum()} train / {tst.sum()} test molecules have finite features")
                if not test_rows:
                    base, _ = fingerprint_shift(fp_full_all[full].astype(np.uint8), fp_test_all[tst].astype(np.uint8))
                    test_rows.append((f"{_BASELINE} Tanimoto (baseline)", base))
                test_metrics = evaluate_shift(
                    X_full_all[full], X_test_all[tst], max_samples=args.max_samples, seed=args.seed,
                )
                test_dropped = int((~tst).sum())
                test_rows.append((name + (f" (-{test_dropped} non-finite)" if test_dropped else ""), test_metrics))
        except Exception as exc:  # one failing representation should not stop the rest
            print(f"[{name}] failed: {exc}", file=sys.stderr)
            continue
        dropped = int((~tr).sum() + (~va).sum())
        rows.append((name + (f" (-{dropped} non-finite)" if dropped else ""), metrics))

    # separate tables so each fits an 80-column terminal without squeezing the names
    shift_columns = [("NN ratio", "nn_ratio", 2), ("Out", "frac_out", 3), ("AUC", "two_sample_auc", 3), ("MMD p", "mmd_p", 3)]
    groups = {
        "Embedding quality": (
            [
                ("Dim", "dim", 0), ("Probe R2", "probe_r2", 3), ("kNN R2", "knn_r2", 3),
                ("Recall", "recall@10", 3), ("EffRank", "eff_rank", 1), ("Rough", "roughness", 3),
            ],
            rows,
        ),
        "Train to val distance": ([*shift_columns, ("Err rho", "err_dist_rho", 3)], rows),
    }
    if args.test:
        groups["Train to test distance (test.csv vs all of train.csv)"] = (shift_columns, test_rows)
    for title, (columns, group_rows) in groups.items():
        table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", pad_edge=False, title=title, title_style="bold")
        table.add_column("Representation", style="bold", overflow="fold", min_width=12)
        for header, _, _ in columns:
            table.add_column(header, justify="right", no_wrap=True)
        for label, m in group_rows:
            is_baseline = label.endswith("(baseline)")
            table.add_row(
                f"[dim]{escape(label)}[/]" if is_baseline else escape(label),
                *(_fmt(m.get(key, float("nan")), digits) for _, key, digits in columns),
            )
        console.print()
        console.print(table)
    console.print(f"\n[dim]{_METRIC_LEGEND}[/]")


def main() -> None:
    """Entry point for the representations CLI."""
    parser = argparse.ArgumentParser(description="Inspect, fill, or evaluate the representation feature cache.")
    parser.add_argument("--check", action="store_true", help="Show which representations are cached and how complete they are")
    parser.add_argument(
        "--evaluate",
        nargs="+",
        metavar="REPRESENTATION",
        help="Score these representations (a registered name, chemprop[/path/to/weights], or 'all') on a "
        "train/val split: probe and kNN R2, Tanimoto-neighbour recall, effective rank, roughness, "
        "train-to-val NN distance, train-vs-val AUC, MMD, and error vs distance. Uses the cache, computing what is missing.",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --evaluate, also measure the blinded data/test.csv against all of train.csv "
        "(distance metrics only: NN ratio, Out, AUC, MMD p; no labels are needed). Default on; --no-test skips it",
    )
    parser.add_argument("--split", default="scaffold", choices=["random", "scaffold", "butina"], help="Split for --evaluate (default: scaffold)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction for --evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the split and the subsampling in --evaluate")
    parser.add_argument("--butina-cutoff", type=float, default=0.4, help="Tanimoto distance cutoff for --split butina")
    parser.add_argument("--targets", nargs="+", help="Label columns for --evaluate (default: every *_pIC50_direct_inhibition)")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Cap on molecules used by the pairwise metrics (recall, rank, AUC) in --evaluate",
    )
    parser.add_argument(
        "--embed",
        nargs="+",
        metavar="REPRESENTATION",
        help="Compute and cache these representations for the train and test molecules "
        "(a registered name, chemprop[/path/to/weights], unimol[/path/to/model_0.pth], or 'all'). Only uncached molecules are computed.",
    )
    parser.add_argument(
        "--alias",
        nargs=2,
        metavar=("PATH", "ALIAS"),
        help=f"Give a foundation-model weights path a short alias, usable as chemprop[ALIAS] or unimol[ALIAS]. Saved in {ALIAS_FILE}.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train.csv and test.csv")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/features"), help="Directory for cached feature matrices")
    args = parser.parse_args()

    if not args.check and not args.embed and not args.alias and not args.evaluate:
        parser.error("pass --check, --embed, --alias and/or --evaluate")
    if args.evaluate and not 0.0 < args.val_split < 1.0:
        parser.error(f"--val-split must be in (0, 1), got {args.val_split}")

    if args.alias:
        path, alias = args.alias
        if not Path(path).expanduser().is_file():
            print(f"Warning: {path} does not exist (alias saved anyway)", file=sys.stderr)
        try:
            set_alias(path, alias)
        except ValueError as exc:
            parser.error(str(exc))
        kind = "unimol" if Path(path).suffix == ".pth" else "chemprop"
        console.print(f"[green]✓[/] {escape(f'{kind}[{alias}]')} -> {escape(str(Path(path).expanduser().resolve()))}", soft_wrap=True)

    if args.embed:
        names = list(INPUT_REGISTRY) if args.embed == ["all"] else args.embed
        unknown = [n for n in names if not is_known_input(n)]
        if unknown:
            print(f"Unknown representation(s): {unknown}. Available: {list(INPUT_REGISTRY)} or chemprop[/path] or unimol[/path]", file=sys.stderr)
            sys.exit(1)
        molecules = _molecules(args.data_dir)
        console.print(f"Embedding {len(molecules)} train+test molecules")
        for name in names:
            try:
                features = featurize(molecules, [name], args.cache_dir)
            except Exception as exc:  # one failing representation should not stop the rest
                print(f"[{name}] failed: {exc}", file=sys.stderr)
                continue
            bad = int((~np.isfinite(features)).any(axis=1).sum())
            console.print(f"[green]✓[/] {escape(name)}: {features.shape[1]} dims" + (f", [yellow]{bad} molecules with non-finite values[/]" if bad else ""))

    if args.evaluate:
        _evaluate(args)

    if args.check:
        smiles = set(_molecules(args.data_dir)["SMILES"].to_list()) if (args.data_dir / "train.csv").exists() else None
        _check(args.cache_dir, smiles)


if __name__ == "__main__":
    main()
