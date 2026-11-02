"""CLI for generating a submission CSV from a trained CYP model."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from data_tools.inputs import INPUT_REGISTRY, featurize, is_known_input
from models import REGISTRY
from models.utils import drop_nan_rows

_CYP_TARGETS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]


def _generate_separate(
    train_path: Path,
    test_path: Path,
    model_name: str,
    input_names: list[str],
    targets: list[str],
    output_path: Path,
    cache_dir: Path,
) -> pl.DataFrame:
    """Fit one model per isoform (sharing one feature matrix), predict on test, write one combined submission CSV."""
    if model_name not in REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {list(REGISTRY.keys())}")
    unknown = [n for n in input_names if not is_known_input(n)]
    if unknown:
        raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}")

    train_df = pl.read_csv(train_path)
    unknown_targets = [t for t in targets if t not in train_df.columns]
    if unknown_targets:
        raise ValueError(f"Target(s) {unknown_targets} not found in train data. Columns: {list(train_df.columns)}")

    test_df = pl.read_csv(test_path)

    # Features don't depend on the target isoform, so compute once and reuse across isoforms.
    train_features = featurize(train_df, input_names, cache_dir)
    X_test_raw = featurize(test_df, input_names, cache_dir)
    valid_mask = ~np.isnan(X_test_raw).any(axis=1)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"Dropped {n_dropped} of {len(test_df)} test molecules with NaN features")
    X_test = X_test_raw[valid_mask]
    test_df_clean = test_df.filter(pl.Series(valid_mask.tolist()))

    result = test_df_clean.select(["Molecule_Name", "SMILES"])
    per_target_train_y: dict[str, np.ndarray] = {}
    for target in targets:
        X_train, y_train = drop_nan_rows(train_features, train_df[target].to_numpy(), label=f"train/{target}")
        model = REGISTRY[model_name]()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        result = result.with_columns(pl.Series(target, preds.tolist()))
        per_target_train_y[target] = y_train
        _print_prediction_summary(preds, target)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output_path)
    print(f"Wrote {len(result)} rows -> {output_path}")
    _plot_prediction_distributions(result, per_target_train_y, targets, output_path)
    return result


def _print_prediction_summary(preds: np.ndarray, target: str) -> None:
    """Print distribution statistics for the generated predictions."""
    p = preds.astype(float)
    percentiles = np.percentile(p, [5, 25, 50, 75, 95])
    print(f"\nPrediction summary ({target}, n={len(p)}):")
    print(f"  mean   : {p.mean():.4f}")
    print(f"  std    : {p.std():.4f}")
    print(f"  min    : {p.min():.4f}")
    print(f"  p5     : {percentiles[0]:.4f}")
    print(f"  p25    : {percentiles[1]:.4f}")
    print(f"  median : {percentiles[2]:.4f}")
    print(f"  p75    : {percentiles[3]:.4f}")
    print(f"  p95    : {percentiles[4]:.4f}")
    print(f"  max    : {p.max():.4f}")


def _plot_prediction_distributions(
    result: pl.DataFrame,
    per_target_train_y: dict[str, np.ndarray],
    targets: list[str],
    output_path: Path,
) -> None:
    """Save a grid of overlaid density histograms, one subplot per isoform."""
    n = len(targets)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for i, target in enumerate(targets):
        ax = axes[i // ncols][i % ncols]
        p = result[target].to_numpy().astype(float)
        t = per_target_train_y[target].astype(float)
        all_vals = np.concatenate([p, t])
        bins = np.linspace(all_vals.min(), all_vals.max(), 31)
        ax.hist(t, bins=bins, density=True, alpha=0.5, label=f"train (n={len(t)}, mean={t.mean():.2f})")
        ax.hist(p, bins=bins, density=True, alpha=0.5, label=f"predicted (n={len(p)}, mean={p.mean():.2f})")
        ax.axvline(t.mean(), color="C0", linestyle="--", linewidth=1.2)
        ax.axvline(p.mean(), color="C1", linestyle="--", linewidth=1.2)
        ax.set_xlabel(target)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.tight_layout()
    plot_path = output_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved distribution plot -> {plot_path}")


def main() -> None:
    """Entry point for the generate-results CLI."""
    parser = argparse.ArgumentParser(description="Generate a submission CSV from a trained CYP model.")
    parser.add_argument("--train-path", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--test-path", default="data/test.csv", help="Path to blinded test CSV")
    parser.add_argument(
        "--model",
        default="decision_tree",
        help=f"Model name to use. Available: {list(REGISTRY.keys())}",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["rdkit"],
        help=f"Input featurization(s) to use (hstacked if multiple). Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--isoform-mode",
        default="separate",
        choices=["separate"],
        help="How to handle the 4 CYP isoforms. 'separate' (default): fit an independent "
        "model per isoform, sharing one feature matrix, and write one combined submission CSV.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=_CYP_TARGETS,
        choices=_CYP_TARGETS,
        help="CYP target column(s) to predict (default: all four isoforms)",
    )
    parser.add_argument("--output", default=None, help="Output CSV path (default: results/<model>_<input>_submission.csv)")
    parser.add_argument("--cache-dir", default="data/features", help="Directory for cached feature matrices")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    input_tag = "+".join(args.input)
    default_name = f"{args.model}_{input_tag}_submission.csv"
    output = Path(args.output) if args.output else Path("results") / default_name

    try:
        _generate_separate(
            Path(args.train_path),
            Path(args.test_path),
            args.model,
            args.input,
            args.targets,
            output,
            cache_dir,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
