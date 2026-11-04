"""CLI that draws the evaluate-models diagnostic plots from a saved predictions CSV."""

import argparse
import sys
from pathlib import Path

from models.plots import PLOT_KINDS, rows_from_predictions, show_terminal, usable_kinds, write_files


def main() -> None:
    """Entry point for the plot-results CLI."""
    parser = argparse.ArgumentParser(
        description="Plot results saved by `evaluate-models --predictions-out`; same plots as its --plots flag."
    )
    parser.add_argument("predictions", type=Path, help="Predictions CSV written by evaluate-models --predictions-out")
    parser.add_argument(
        "--plots", nargs="+", choices=PLOT_KINDS, default=PLOT_KINDS,
        help="Plots to draw (default: all that the inputs allow). heldout-skew needs only the predictions CSV; "
        "train-test also needs the blind-set predictions (the .blind.csv written next to --predictions-out)",
    )
    parser.add_argument("--plots-dir", type=Path, default=Path("results/plots"), help="Directory for the PNGs")
    parser.add_argument("--plots-terminal", action="store_true", help="Also draw the plots in the terminal")
    parser.add_argument(
        "--blind-predictions", type=Path,
        help="Blind-set predictions CSV for train-test (default: <predictions>.blind.csv next to the input)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing data CSVs")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction used in the run")
    parser.add_argument("--butina-cutoff", type=float, default=0.4, help="Butina cutoff used in the run")
    args = parser.parse_args()

    blind = args.blind_predictions or args.predictions.with_suffix(".blind.csv")
    try:
        rows = rows_from_predictions(args.predictions, blind, args.data_dir, args.val_split, args.butina_cutoff)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot read {args.predictions}: {exc}", file=sys.stderr)
        sys.exit(1)
    kinds = usable_kinds(args.plots, rows)
    if not kinds:
        sys.exit(1)
    written = write_files(kinds, rows, args.plots_dir)
    print(f"Wrote {len(written)} plots to {args.plots_dir}", file=sys.stderr)
    if args.plots_terminal:
        show_terminal(kinds, rows)


if __name__ == "__main__":
    main()
