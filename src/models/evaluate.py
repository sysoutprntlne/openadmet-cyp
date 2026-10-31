"""CLI for evaluating CYP challenge models on a train/validation split."""

import argparse
import csv
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from data_tools.inputs import INPUT_REGISTRY, featurize, is_known_input
from data_tools.load import cv_folds, load_data
from models import REGISTRY, CYPModel
from models.utils import drop_nan_rows, zero_nonfinite_features

_CYP_TARGETS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

_ISOFORMS = {t.split("_")[0].removeprefix("CYP"): t for t in _CYP_TARGETS}

console = Console()

_PREDICTION_COLUMNS = ["smiles", "isoform", "model", "input", "split", "seed", "y_true", "y_pred"]


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a new CSV, rejecting ambiguous join keys before creating the file."""
    seen = set()
    for row in rows:
        key = tuple(row[column] for column in _PREDICTION_COLUMNS[:6])
        if key in seen:
            raise ValueError(
                f"Cannot export predictions: duplicate SMILES {row['smiles']!r} within "
                f"isoform={row['isoform']}, model={row['model']}, input={row['input']}, "
                f"split={row['split']}, seed={row['seed']}; downstream SMILES joins would be ambiguous. "
                "No rows were deduplicated or written."
            )
        seen.add(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_PREDICTION_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise OSError(f"Cannot write predictions CSV to {path}: {exc}") from exc


_MIN_LABEL_WIDTH = 4
_METRICS = ["RMSE", "MAE", "R2", "ST-RAE", "Spearman", "Kendall"]
# Direction each metric improves in: True = lower is better.
_HEADERS = {"RMSE": "RMSE", "MAE": "MAE", "R2": "R2", "ST-RAE": "ST-RAE", "Spearman": "ρ", "Kendall": "τ"}
_LOWER_IS_BETTER = {"RMSE": True, "MAE": True, "R2": False, "ST-RAE": True, "Spearman": False, "Kendall": False}


def _compute_metrics(actual: pl.Series, predicted: pl.Series) -> dict[str, float]:
    """Compute RMSE, MAE, R2, ST-RAE, Spearman's rho and Kendall's tau from actual and predicted Series.

    ST-RAE is the relative absolute error of one target: total absolute error over
    the total absolute error of always predicting the validation mean (<1 beats the mean).
    Rank correlations are NaN when either side is constant.
    """
    a = np.array([float(v) for v in actual])
    p = np.array([float(v) for v in predicted])
    n = len(a)
    ss_res = float(((a - p) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    abs_err = float(np.abs(a - p).sum())
    abs_dev = float(np.abs(a - a.mean()).sum())
    constant = a.std() == 0 or p.std() == 0
    return {
        "RMSE": (ss_res / n) ** 0.5,
        "MAE": abs_err / n,
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0,
        "ST-RAE": abs_err / abs_dev if abs_dev > 0.0 else float("nan"),
        "Spearman": float("nan") if constant else float(stats.spearmanr(a, p).statistic),
        "Kendall": float("nan") if constant else float(stats.kendalltau(a, p).statistic),
    }


def _evaluate_model(
    model: CYPModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    """Fit and predict once; optionally return the predictions used for metrics."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    preds = pl.Series("prediction", predictions.tolist())
    metrics = _compute_metrics(pl.Series("actual", y_val.tolist()), preds)
    return (metrics, predictions) if return_predictions else metrics


_ENSEMBLE_METHODS = {"average": np.mean, "min": np.min, "max": np.max}


def _combine(predictions: list[np.ndarray], method: str) -> np.ndarray:
    """Reduce a list of equal-length prediction vectors to one with the ensemble method."""
    return _ENSEMBLE_METHODS[method](np.stack(predictions), axis=0)


def _fit_members(
    cls: type[CYPModel],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_blind: np.ndarray | None,
    folds: np.ndarray | None,
    seeds: list[int | None],
) -> dict:
    """Fit one model per (seed, fold) and collect what each predicts.

    Without folds each seed is fit on all training rows. With folds each model is fit on the
    other folds and also predicts its own held-out fold. A seed of None keeps the model's
    default; models without a `random_state` argument are fit once, since seeds can't differ.
    """
    if "random_state" not in inspect.signature(cls.__init__).parameters:
        seeds = [None]
    members: dict = {"val": [], "blind": [], "holdout": {}}
    for seed in seeds:
        for k in ([None] if folds is None else np.unique(folds).tolist()):
            model = cls() if seed is None else cls(random_state=seed)
            fit_rows = np.ones(len(y_train), dtype=bool) if k is None else folds != k
            model.fit(X_train[fit_rows], y_train[fit_rows])
            members["val"].append(model.predict(X_val))
            if X_blind is not None:
                members["blind"].append(model.predict(X_blind))
            if k is not None:
                members["holdout"].setdefault(k, []).append(model.predict(X_train[folds == k]))
    return members


def _collapse(members: list[dict], method: str) -> dict:
    """Combine member predictions (over seeds and folds, or over models) into one prediction set."""
    return {
        "val": _combine([m for g in members for m in g["val"]], method),
        "blind": _combine([m for g in members for m in g["blind"]], method) if members[0]["blind"] else None,
        "holdout": {k: _combine([m for g in members for m in g["holdout"][k]], method) for k in members[0]["holdout"]},
    }


def _cv_metrics(holdout: dict[int, np.ndarray], y: np.ndarray, folds: np.ndarray) -> dict[str, float]:
    """Score each fold's held-out predictions and mean every metric over folds."""
    per_fold = [
        _compute_metrics(pl.Series("actual", y[folds == k].tolist()), pl.Series("prediction", pred.tolist()))
        for k, pred in sorted(holdout.items())
    ]
    sizes = {"n_train": len(y) - len(y) / len(per_fold), "n_val": len(y) / len(per_fold)}
    return {
        **{
            m: float(np.nanmean([f[m] for f in per_fold])) if not all(np.isnan(f[m]) for f in per_fold) else float("nan")
            for m in _METRICS
        },
        **{k: round(v) for k, v in sizes.items()},
    }


# Looser than validation.r_squared_filtering (0.3) in the run-ensemble pipeline.
_CV_FILTER_R2 = 0.15


def _oof_r2(holdout: dict[int, np.ndarray], y: np.ndarray, folds: np.ndarray) -> float:
    """R² of the out-of-fold predictions pooled over all folds."""
    oof = np.empty(len(y))
    for k, pred in holdout.items():
        oof[folds == k] = pred
    return _compute_metrics(pl.Series("actual", y.tolist()), pl.Series("prediction", oof.tolist()))["R2"]


def _resolve_models(names: list[str]) -> list[type[CYPModel]]:
    """Return model classes for the given names, or all registered models if names == ['all']."""
    if names == ["all"]:
        return list(REGISTRY.values())
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(REGISTRY.keys())}")
    return [REGISTRY[n] for n in names]


def _resolve_inputs(groups: list[list[str]]) -> list[list[str]]:
    """Return the input sets to evaluate: each registered input alone for ['all'], else each group hstacked."""
    sets: list[list[str]] = []
    for names in groups:
        if names == ["all"]:
            sets.extend([n] for n in INPUT_REGISTRY)
            continue
        unknown = [n for n in names if not is_known_input(n)]
        if unknown:
            raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}")
        sets.append(names)
    return sets


def _short(label: str) -> str:
    """Shorten `chemprop[/long/path/best.pt]` to `chemprop[best.pt]` for display."""
    return re.sub(r"(chemprop|unimol)\[([^\]]*)\]", lambda m: f"{m.group(1)}[{Path(m.group(2)).name}]", label)


def _ellipsize(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(width - 1, 1)] + "…"


def _aggregate(results: list[tuple[str, str, str, dict[str, float]]]) -> list[tuple[str, str, dict[str, float]]]:
    """Macro-average each metric over isoforms per (model, input); NaNs are skipped per metric."""
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for _, name, inp, m in results:
        grouped.setdefault((name, inp), []).append(m)
    return [
        (name, inp, {k: float(np.nanmean([m[k] for m in ms])) if not all(np.isnan(m[k]) for m in ms) else float("nan")
                     for k in _METRICS})
        for (name, inp), ms in grouped.items()
    ]


def _format_cell(metric: str, value: float, best: float) -> str:
    if np.isnan(value):
        return "[dim]n/a[/]"
    text = f"{value:.3f}"
    if value == best:
        return f"[bold green]{text}[/]"
    if metric in ("R2", "Spearman", "Kendall") and value < 0:
        return f"[red]{text}[/]"
    return text


def _print_results(results: list[tuple[str, str, str, dict[str, float]]], sort_by: str) -> None:
    """Print one metrics table with a block per isoform and a final macro-average block.

    The best value on each metric within a block is bold green.
    """
    by_isoform: dict[str, list[tuple[str, str, dict[str, float]]]] = {}
    for iso, name, inp, m in results:
        by_isoform.setdefault(iso, []).append((name, inp, m))
    blocks = list(by_isoform.items())
    if len(by_isoform) > 1:
        macro = _aggregate(results)
        reverse = not _LOWER_IS_BETTER[sort_by]
        macro.sort(key=lambda r: (-r[2][sort_by] if reverse else r[2][sort_by]) if not np.isnan(r[2][sort_by]) else float("inf"))
        blocks.append(("MEAN", macro))

    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", title_style="bold", padding=(0, 1), pad_edge=False)
    table.add_column("Isoform", style="bold", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Input", no_wrap=True)
    # Data-point counts ride along in each result's metrics dict; the MEAN rows have none (counts differ per isoform).
    counts = {id(m): f"{int(m['n_train'])}/{int(m['n_val'])}" for _, rows in blocks for _, _, m in rows if "n_train" in m}
    counts_width = max([len("Train/Val"), *map(len, counts.values())])
    table.add_column("Train/Val", justify="right", no_wrap=True, overflow="ignore", min_width=counts_width)
    numeric_width = counts_width + 3
    for metric in _METRICS:
        arrow = " ▼" if _LOWER_IS_BETTER[metric] else " ▲"
        header = _HEADERS[metric] + (arrow if metric == sort_by else "")
        # Numbers are never shortened: the column is as wide as its header or longest value (e.g. "-0.123"), whichever is larger.
        value_width = max((len(f"{r[2][metric]:.3f}") for _, rows in blocks for r in rows), default=0)
        width = max(len(header), value_width)
        numeric_width += width + 3
        table.add_column(header, justify="right", no_wrap=True, overflow="ignore", min_width=width)

    # Only the Model/Input strings are shortened (with "…") to fit the terminal; the numeric columns above keep full width.
    isoform_width = max(len("Isoform"), *(len(iso.replace("_pIC50_direct_inhibition", "")) for iso, _ in blocks))
    label_budget = max(2 * _MIN_LABEL_WIDTH, console.width - numeric_width - isoform_width - 3 * 3 - 2)
    model_width = max(len(r[0]) for _, rows in blocks for r in rows)
    input_width = max(len(_short(r[1])) for _, rows in blocks for r in rows)
    if model_width + input_width > label_budget:
        half = label_budget // 2
        if model_width <= half:
            input_width = label_budget - model_width
        elif input_width <= label_budget - half:
            model_width = label_budget - input_width
        else:
            model_width, input_width = label_budget - half, half

    for i, (iso, rows) in enumerate(blocks):
        best = {}
        for metric, lower in _LOWER_IS_BETTER.items():
            values = [r[2][metric] for r in rows if not np.isnan(r[2][metric])]
            best[metric] = (min if lower else max)(values) if values else float("nan")
        for j, (name, inp, m) in enumerate(rows):
            label = iso.replace("_pIC50_direct_inhibition", "") if j == 0 else ""
            table.add_row(
                f"[bold magenta]{label}[/]" if iso == "MEAN" else label,
                escape(_ellipsize(name, model_width)),
                escape(_ellipsize(_short(inp), input_width)),
                counts.get(id(m), ""),
                *(_format_cell(metric, m[metric], best[metric]) for metric in _METRICS),
                end_section=j == len(rows) - 1 and i < len(blocks) - 1,
            )
    console.print(table)


def _print_correlations(target: str, preds: dict[tuple[str, str], dict[int, float]]) -> None:
    """Print an all-vs-all Pearson correlation matrix of validation predictions for one target.

    Each config's predictions are keyed by validation row index, so configs whose
    NaN filtering dropped different rows are compared on their shared molecules.
    """
    labels = list(preds)
    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        title=f"Prediction correlation (Pearson) · {target.replace('_pIC50_direct_inhibition', '')}",
        title_style="bold",
    )
    table.add_column("", style="bold", overflow="fold")
    for i in range(len(labels)):
        table.add_column(str(i), justify="right")
    for i, a in enumerate(labels):
        cells = []
        for j, b in enumerate(labels):
            shared = sorted(preds[a].keys() & preds[b].keys())
            x = np.array([preds[a][k] for k in shared])
            y = np.array([preds[b][k] for k in shared])
            if len(shared) < 2 or x.std() == 0 or y.std() == 0:
                cells.append("[dim]n/a[/]")
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            color = "dim" if i == j else "green" if r >= 0.9 else "yellow" if r >= 0.6 else "red"
            cells.append(f"[{color}]{r:.2f}[/]")
        table.add_row(escape(f"{i}  {a[0]} [{_short(a[1])}]"), *cells)
    console.print()
    console.print(table)


def main() -> None:
    """Entry point for the evaluate-models CLI."""
    parser = argparse.ArgumentParser(description="Evaluate CYP models on a train/val split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing data CSVs")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument(
        "--split",
        default="scaffold",
        choices=["random", "scaffold", "butina"],
        help="Train/val split strategy (default: scaffold)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument(
        "--butina-cutoff",
        type=float,
        default=0.4,
        help="Tanimoto distance cutoff for Butina clustering (default: 0.4).",
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        default=None,
        help="Also run k-fold cross-validation on the training portion and report metrics averaged "
        "over the held-out folds (default: off)",
    )
    parser.add_argument(
        "--cv-split",
        default=None,
        choices=["random", "scaffold", "butina"],
        help="Fold assignment strategy for --nfolds (default: same as --split). "
        "Scaffold and Butina keep each group within one fold.",
    )
    parser.add_argument(
        "--nseeds",
        type=int,
        default=1,
        help="Train each model with this many seeds (--seed, --seed+1, ...) and ensemble them. "
        "Models without a random_state argument are trained once. Default: 1 (the model's own default seed)",
    )
    parser.add_argument(
        "--ensemble",
        default=None,
        choices=list(_ENSEMBLE_METHODS),
        help="How to combine predictions across folds (--nfolds), seeds (--nseeds) and, when several "
        "--models are given, across models, which adds an ensemble row. Default: average whenever "
        "--nfolds or --nseeds is used",
    )
    parser.add_argument(
        "--cv-filter",
        default=None,
        choices=["r2"],
        help="Drop models from the multi-model ensemble row per target when their seed-averaged out-of-fold "
        f"R² is below {_CV_FILTER_R2}. Requires --nfolds and several --models. "
        "If no model passes, all are kept with a warning. Default: no filtering",
    )
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names to evaluate, or 'all'")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=_CYP_TARGETS,
        help="CYP target column(s) to evaluate (default: all four isoforms)",
    )
    parser.add_argument(
        "--isoforms",
        nargs="+",
        default=None,
        choices=list(_ISOFORMS),
        help="Shorthand for --targets: isoform(s) to evaluate, direct inhibition pIC50 only "
        f"(e.g. --isoforms 3A4 2D6). Choices: {list(_ISOFORMS)}",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        action="append",
        help="Input featurization(s) to use (hstacked if multiple), or 'all' to evaluate each registered "
        "input on its own. Repeat the flag to evaluate several input sets separately, "
        f"e.g. --input xtb_solv_v1 --input chemeleon. Default: rdkit. Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--correlation",
        action="store_true",
        help="After evaluating, print an all-vs-all correlation matrix of validation predictions "
        "across every (model, input) combination, per target",
    )
    parser.add_argument("--cache-dir", default="data/features", help="Directory for cached feature matrices")
    parser.add_argument("--sort-by", default="MAE", choices=_METRICS, help="Metric to sort results by")
    parser.add_argument(
        "--predictions-out", type=Path,
        help="Write validation predictions to a new CSV (use results/); creates parents, refuses existing files",
    )
    parser.add_argument(
        "--predict-blind",
        action="store_true",
        help="Also predict the blinded test set (data/test.csv) with each fitted model and write one CSV per "
        "(model, input) to results/<model>_<input>_blind_predictions.csv. Models are fit on the train split only.",
    )
    args = parser.parse_args()
    args.input = args.input or [["rdkit"]]

    if args.targets is not None and args.isoforms is not None:
        parser.error("--targets and --isoforms are mutually exclusive")
    if args.isoforms is not None:
        args.targets = [_ISOFORMS[i] for i in args.isoforms]
    elif args.targets is None:
        args.targets = _CYP_TARGETS

    if args.nfolds is not None and args.nfolds < 2:
        parser.error(f"--nfolds must be >= 2, got {args.nfolds}")
    if args.nfolds is None and args.cv_split is not None:
        parser.error("--cv-split requires --nfolds")
    args.cv_split = args.cv_split or args.split
    if args.cv_filter is not None and args.nfolds is None:
        parser.error("--cv-filter requires --nfolds")
    if args.nseeds < 1:
        parser.error(f"--nseeds must be >= 1, got {args.nseeds}")
    method = args.ensemble or "average"
    seeds: list[int | None] = [None] if args.nseeds == 1 else [args.seed + i for i in range(args.nseeds)]

    if not 0.0 < args.val_split < 1.0:
        print(f"--val-split must be in (0, 1), got {args.val_split}", file=sys.stderr)
        sys.exit(1)

    if args.predictions_out is not None and args.predictions_out.exists():
        print(
            f"Cannot write predictions CSV to {args.predictions_out}: file already exists "
            "(checked before training; choose a new path or delete it)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        model_classes = _resolve_models(args.models)
        input_sets = _resolve_inputs(args.input)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.cv_filter is not None and len(model_classes) < 2:
        print("--cv-filter only affects the multi-model ensemble row; with one model it does nothing", file=sys.stderr)

    cache_dir = Path(args.cache_dir)

    train_df, val_df = load_data(
        split_type=args.split,
        val_fraction=args.val_split,
        seed=args.seed,
        butina_cutoff=args.butina_cutoff,
        data_dir=args.data_dir,
    )
    print(f"load_data(split_type={args.split!r}, seed={args.seed}): {len(val_df)} val / {len(train_df)} train molecules", file=sys.stderr)

    folds = None
    if args.nfolds is not None:
        try:
            folds = cv_folds(
                train_df, n_folds=args.nfolds, split_type=args.cv_split,
                seed=args.seed, butina_cutoff=args.butina_cutoff,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"cv_folds(cv_split={args.cv_split!r}, nfolds={args.nfolds}): fold sizes {np.bincount(folds).tolist()}", file=sys.stderr)

    results: list[tuple[str, str, str, dict[str, float]]] = []
    cv_results: list[tuple[str, str, str, dict[str, float]]] = []
    prediction_rows: list[dict[str, object]] = []
    # target -> (model, input) -> {val row index: prediction}
    corr_preds: dict[str, dict[tuple[str, str], dict[int, float]]] = {}
    blind_df = pl.read_csv(args.data_dir / "test.csv") if args.predict_blind else None
    # (model, input) -> target -> prediction per blind row
    blind_preds: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for input_names in input_sets:
        input_label = " + ".join(input_names)
        # Features don't depend on the target isoform, so compute once and reuse across isoforms.
        train_features = zero_nonfinite_features(
            featurize(train_df, input_names, cache_dir), f"train/{input_label}", train_df["SMILES"].to_list(),
        )
        val_features = zero_nonfinite_features(
            featurize(val_df, input_names, cache_dir), f"val/{input_label}", val_df["SMILES"].to_list(),
        )
        if blind_df is not None:
            blind_features = zero_nonfinite_features(
                featurize(blind_df, input_names, cache_dir), f"blind/{input_label}", blind_df["SMILES"].to_list(),
            )
        for target in args.targets:
            label = f"{target}, {input_label}"
            X_train, y_train, train_mask = drop_nan_rows(
                train_features, train_df[target].to_numpy(), label=f"train/{label}", return_mask=True,
            )
            X_val, y_val, val_mask = drop_nan_rows(
                val_features, val_df[target].to_numpy(), label=f"val/{label}", return_mask=True,
            )
            val_index = np.flatnonzero(val_mask)
            if args.predictions_out is not None:
                val_smiles = val_df["SMILES"].to_numpy()[val_mask]
            fold_ids = None if folds is None else folds[train_mask]
            collapsed = [
                (cls.name, _collapse([_fit_members(
                    cls, X_train, y_train, X_val, blind_features if blind_df is not None else None, fold_ids, seeds,
                )], method))
                for cls in model_classes
            ]
            if len(collapsed) > 1 and (args.ensemble or folds is not None or args.nseeds > 1):
                members = collapsed
                if args.cv_filter == "r2":
                    scores = {name: _oof_r2(c["holdout"], y_train, fold_ids) for name, c in collapsed}
                    members = [(name, c) for name, c in collapsed if scores[name] >= _CV_FILTER_R2]
                    dropped = ", ".join(f"{n} (R²={r:.3f})" for n, r in scores.items() if r < _CV_FILTER_R2)
                    if not members:
                        members = collapsed
                        print(
                            f"cv-filter r2: no model reaches out-of-fold R² >= {_CV_FILTER_R2} for {label}; "
                            f"keeping all {len(collapsed)} in the ensemble ({dropped})",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"cv-filter r2: {label}: kept {len(members)}/{len(collapsed)} models"
                            + (f", dropped {dropped}" if dropped else ""),
                            file=sys.stderr,
                        )
                as_members = [
                    {"val": [c["val"]], "blind": [] if c["blind"] is None else [c["blind"]],
                     "holdout": {k: [v] for k, v in c["holdout"].items()}}
                    for _, c in members
                ]
                collapsed.append((f"ensemble[{method}]", _collapse(as_members, method)))
            for name, out in collapsed:
                predictions = out["val"]
                metrics = _compute_metrics(pl.Series("actual", y_val.tolist()), pl.Series("prediction", predictions.tolist()))
                if args.correlation:
                    corr_preds.setdefault(target, {})[(name, input_label)] = dict(
                        zip(val_index.tolist(), predictions.tolist())
                    )
                if args.predictions_out is not None:
                    prediction_rows.extend(
                        dict(zip(_PREDICTION_COLUMNS, (
                            smiles, target, name, input_label, args.split, args.seed, float(actual), float(pred),
                        )))
                        for smiles, actual, pred in zip(val_smiles, y_val, predictions)
                    )
                if blind_df is not None:
                    blind_preds.setdefault((name, input_label), {})[target] = out["blind"]
                results.append((target, name, input_label, {**metrics, "n_train": len(y_train), "n_val": len(y_val)}))
                if fold_ids is not None:
                    cv_results.append((target, name, input_label, _cv_metrics(out["holdout"], y_train, fold_ids)))

    if blind_df is not None:
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        for (name, input_label), by_target in blind_preds.items():
            path = out_dir / f"{re.sub(r'[^\w+.-]', '_', name)}_{re.sub(r'[^\w+.-]', '_', input_label.replace(' + ', '+'))}_blind_predictions.csv"
            blind_df.select(["Molecule_Name", "SMILES"]).with_columns(
                [pl.Series(target, pred) for target, pred in by_target.items()]
            ).write_csv(path)
            print(f"Wrote {len(blind_df)} blind predictions to {path}", file=sys.stderr)

    if args.predictions_out is not None:
        try:
            _write_predictions(args.predictions_out, prediction_rows)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Wrote {len(prediction_rows)} validation predictions to {args.predictions_out}", file=sys.stderr)

    reverse = not _LOWER_IS_BETTER[args.sort_by]

    def _sort_key(r: tuple[str, str, str, dict[str, float]]) -> tuple[str, float]:
        value = r[3][args.sort_by]
        return (r[0], float("inf") if np.isnan(value) else -value if reverse else value)

    results.sort(key=_sort_key)
    cv_results.sort(key=_sort_key)
    console.print(
        f"[dim]input[/]="
        + escape(_short(" | ".join(" + ".join(g) for g in args.input)))
        + f"  [dim]split[/]={args.split}  [dim]seed[/]={args.seed}"
        + (f"  [dim]nseeds[/]={args.nseeds}" if args.nseeds > 1 else "")
        + (f"  [dim]ensemble[/]={method}" if args.ensemble or folds is not None or args.nseeds > 1 else "")
        + (f"  [dim]cv-filter[/]={args.cv_filter}" if args.cv_filter else "")
    )
    _print_results(results, args.sort_by)
    if cv_results:
        console.print(
            f"\n[dim]cross-validation[/]  [dim]cv-split[/]={args.cv_split}  [dim]nfolds[/]={args.nfolds}"
            "  [dim](metrics averaged over held-out folds of the training portion)[/]"
        )
        _print_results(cv_results, args.sort_by)
    if args.correlation:
        for target, preds in corr_preds.items():
            if len(preds) < 2:
                print(f"\n--correlation needs at least two (model, input) combinations; got {len(preds)}", file=sys.stderr)
                break
            _print_correlations(target, preds)


if __name__ == "__main__":
    main()
