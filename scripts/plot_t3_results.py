"""Draw the T3 (model zoo) charts from `evaluate-models` output.

Save each evaluator run to a text file, e.g.

    evaluate-models --models ridge xgboost --input rdkit --split scaffold 2>&1 | tee reports/results_rdkit.txt

then, from the repo root with the venv active:

    python scripts/plot_t3_results.py reports/results_rdkit.txt reports/results_morgan.txt
    python scripts/plot_t3_results.py reports/results_*.txt --union data/union_train.csv --out reports

Writes into --out (default: reports/):
    model_rmse_by_isoform.png    RMSE of every model, one panel per isoform
    best_model_r2.png            R^2 of the lowest-RMSE model for each isoform
    data_volume_per_isoform.png  pIC50 labels per isoform and source (needs --union)
    t3_results_long.csv          every parsed result row, so the numbers can be checked

Nothing is typed in by hand: every number on a chart comes from the files you pass in.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# One results row of the evaluator table: isoform, model, input, RMSE, MAE, R2.
ROW = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)$")
# "Dropped 2832 of 3924 examples ... (train/CYP1A2_pIC50_direct_inhibition, rdkit)"
DROPPED = re.compile(r"Dropped (\d+) of (\d+) examples.*\((train|val)/(\S+?),\s*(\S+?)\)")

MODEL_ORDER = ["random_forest", "lightgbm", "xgboost", "tabpfn", "ridge", "decision_tree"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SOURCE_NAMES = {
    "aid1851": "PubChem AID 1851 (qHTS)",
    "direct_inhibition": "Challenge (direct inhibition)",
}

plt.rcParams.update(
    {
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GRID,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
    }
)


def short(isoform):
    """CYP1A2_pIC50_direct_inhibition -> CYP1A2."""
    return isoform.split("_")[0]


def parse(paths):
    """Return (result rows, {(isoform, 'train'|'val'): examples kept after dropping NaNs})."""
    rows = {}
    kept = {}
    for path in paths:
        text = Path(path).read_text(errors="replace")
        for line in re.split(r"[\r\n]+", text):
            line = line.strip()
            m = ROW.match(line)
            if m and m.group(1).startswith("CYP"):
                key = (m.group(1), m.group(2), m.group(3))
                rows[key] = {
                    "isoform": key[0],
                    "model": key[1],
                    "input": key[2],
                    "rmse": float(m.group(4)),
                    "mae": float(m.group(5)),
                    "r2": float(m.group(6)),
                }
                continue
            d = DROPPED.search(line)
            if d:
                kept.setdefault((d.group(4), d.group(3)), int(d.group(2)) - int(d.group(1)))
    return list(rows.values()), kept


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def plot_rmse(rows, out, cap):
    isoforms = sorted({r["isoform"] for r in rows})
    inputs = list(dict.fromkeys(r["input"] for r in rows))
    if len(inputs) > len(SERIES):
        sys.exit(f"Too many feature sets for one chart ({len(inputs)}); pass at most {len(SERIES)}.")
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    models += sorted({r["model"] for r in rows} - set(models))
    rmse = {(r["isoform"], r["model"], r["input"]): r["rmse"] for r in rows}
    colors = dict(zip(inputs, SERIES))
    height = min(0.34, 0.8 / len(inputs))
    top = min(cap, max(r["rmse"] for r in rows))
    clipped = any(r["rmse"] > cap for r in rows)
    xmax = top * (1.05 if clipped else 1.2)

    fig, axes = plt.subplots(
        1, len(isoforms), figsize=(3.3 * len(isoforms) + 0.5, 4.6), dpi=200,
        sharey=True, squeeze=False,
    )
    for ax, iso in zip(axes[0], isoforms):
        for j, inp in enumerate(inputs):
            offset = (j - (len(inputs) - 1) / 2) * (height + 0.02)
            present = [m for m in models if (iso, m, inp) in rmse]
            for m in present:
                v = rmse[(iso, m, inp)]
                y = models.index(m) + offset
                ax.barh(y, min(v, cap), height=height, color=colors[inp])
                if v > cap:
                    ax.text(cap - 0.02, y, f"{v:.2f} →", ha="right", va="center",
                            fontsize=8, color="white")
            if present:  # label only the best bar of each feature set
                best = min(present, key=lambda m: rmse[(iso, m, inp)])
                v = rmse[(iso, best, inp)]
                ax.text(v + 0.02, models.index(best) + offset, f"{v:.3f}",
                        va="center", fontsize=8.5)
        ax.set_title(short(iso), fontsize=12, fontweight="bold", loc="left")
        ax.set_xlim(0, xmax)
        ax.set_yticks(range(len(models)), [m.replace("_", " ") for m in models])
        ax.invert_yaxis()
        ax.xaxis.grid(True, color=GRID, lw=1)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("RMSE (lower = better)")
        style_axes(ax)
    if len(inputs) > 1:
        fig.legend(
            handles=[Patch(color=colors[i], label=i) for i in inputs],
            loc="upper right", ncol=len(inputs), frameon=False, fontsize=10,
            bbox_to_anchor=(0.99, 1.0), title="features", title_fontsize=9,
        )
    fig.suptitle("Model comparison by isoform", x=0.01, ha="left", fontsize=14,
                 fontweight="bold")
    note = "Source: evaluate-models output files. Best bar per feature set is labeled."
    if clipped:
        note += f" Bars above {cap:g} are clipped and labeled."
    fig.text(0.01, 0.005, note, fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out / "model_rmse_by_isoform.png")
    plt.close(fig)


def plot_best_r2(rows, kept, out):
    isoforms = sorted({r["isoform"] for r in rows})
    best = {i: min((r for r in rows if r["isoform"] == i), key=lambda r: r["rmse"])
            for i in isoforms}
    labels = []
    for i in isoforms:
        b = best[i]
        label = f"{short(i)}\n{b['model'].replace('_', ' ')} + {b['input']}"
        if (i, "train") in kept and (i, "val") in kept:
            label += f"\nn = {kept[(i, 'train')]:,} train / {kept[(i, 'val')]:,} val"
        labels.append(label)
    r2 = [best[i]["r2"] for i in isoforms]

    fig, ax = plt.subplots(figsize=(2 + 1.7 * len(isoforms), 4.8), dpi=200)
    ax.bar(range(len(isoforms)), r2, width=0.5, color=SERIES[0])
    pad = 0.015
    for x, v in enumerate(r2):
        ax.text(x, v + pad if v >= 0 else v - pad, f"R² = {v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(isoforms)), labels, fontsize=9.5)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(min(0, min(r2)) - 0.06, max(r2) + 0.12)
    ax.set_ylabel("R² on validation set")
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.axhline(0, color=INK2, lw=1)
    style_axes(ax)
    ax.set_title("Best model per isoform (lowest validation RMSE)", loc="left", fontsize=13,
                 fontweight="bold")
    fig.text(0.01, 0.01, "R² = 0 means no better than guessing the mean. "
             "Source: evaluate-models output files.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "best_model_r2.png")
    plt.close(fig)
    return best


def plot_volume(union_path, out):
    import polars as pl  # only needed for this chart

    df = pl.read_csv(union_path, infer_schema_length=None)
    pattern = re.compile(r"^(CYP[0-9A-Z]+)_pIC50_(.+)$")
    counts = defaultdict(dict)
    for col in df.columns:
        m = pattern.match(col)
        if m:
            counts[m.group(1)][m.group(2)] = df.height - df[col].null_count()
    if not counts:
        print("No <gene>_pIC50_<source> columns found in the union table; skipping volume chart.")
        return
    genes = sorted(counts)
    sources = [s for s in SOURCE_NAMES if any(s in counts[g] for g in genes)]
    sources += sorted({s for g in genes for s in counts[g]} - set(sources))
    sources = sources[: len(SERIES)]

    print(f"\npIC50 labels per isoform in {union_path} ({df.height:,} rows):")
    for g in genes:
        print(f"  {g}: " + ", ".join(f"{s}={counts[g].get(s, 0):,}" for s in sources))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bottoms = [0] * len(genes)
    totals = [sum(counts[g].get(s, 0) for s in sources) for g in genes]
    ymax = max(totals) * 1.18
    for k, s in enumerate(sources):
        vals = [counts[g].get(s, 0) for g in genes]
        ax.bar(genes, vals, bottom=bottoms, color=SERIES[k], width=0.7,
               label=SOURCE_NAMES.get(s, s))
        for x, (v, b) in enumerate(zip(vals, bottoms)):
            if v > ymax * 0.07:  # label inside only when the segment is tall enough
                ax.text(x, b + v / 2, f"{v:,}", ha="center", va="center", color="white",
                        fontsize=10)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, t in enumerate(totals):
        ax.text(x, t + ymax * 0.01, f"{t:,}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Labeled molecules (pIC50)")
    ax.set_title(f"pIC50 labels per CYP isoform, by data source\n(union table, "
                 f"{df.height:,} molecules)", fontsize=12)
    ax.legend(frameon=False, loc="upper center", ncol=len(sources), fontsize=8)
    style_axes(ax)
    fig.text(0.01, 0.01, f"Source: non-null counts in {Path(union_path).name}. "
             "pctinhib readouts are not shown.", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "data_volume_per_isoform.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", type=Path, help="saved evaluate-models outputs")
    parser.add_argument("--union", type=Path, help="data/union_train.csv (data-volume chart)")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="output folder")
    parser.add_argument("--cap", type=float, default=1.3, help="clip RMSE bars above this")
    args = parser.parse_args()

    rows, kept = parse(args.results)
    if not rows:
        sys.exit("No result rows found. Save the evaluator's output with "
                 "`evaluate-models ... 2>&1 | tee reports/results_<name>.txt`.")
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "t3_results_long.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["isoform", "model", "input", "rmse", "mae", "r2"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["isoform"], r["rmse"])))

    plot_rmse(rows, args.out, args.cap)
    best = plot_best_r2(rows, kept, args.out)
    if args.union:
        plot_volume(args.union, args.out)

    print(f"Parsed {len(rows)} result rows from {len(args.results)} file(s).")
    print("Lowest RMSE per isoform:")
    for iso, r in best.items():
        print(f"  {short(iso)}: {r['model']} + {r['input']}  RMSE {r['rmse']:.4f}  R2 {r['r2']:.4f}")
    print(f"Charts written to {args.out}/")


if __name__ == "__main__":
    main()