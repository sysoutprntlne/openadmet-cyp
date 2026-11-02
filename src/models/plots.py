"""Diagnostic plots for `evaluate-models --plots`: PNG files and an in-terminal grid."""

import base64
import csv
import io
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PLOT_KINDS = ["train-test", "heldout-skew"]


def _slug(text: str) -> str:
    return re.sub(r"[^\w+.-]", "_", text.replace(" + ", "+"))


def _skew_stats(y_val: np.ndarray, pred_val: np.ndarray) -> tuple[float, float, float]:
    """Mean bias (pred - exp) and the (slope, intercept) of pred regressed on exp."""
    bias = float(np.mean(pred_val - y_val))
    if len(y_val) > 1 and np.ptp(y_val) > 0:
        slope, intercept = np.polyfit(y_val, pred_val, 1)
        return bias, float(slope), float(intercept)
    return bias, float("nan"), float("nan")


def draw_train_test(ax, title: str, r: dict, compact: bool = False) -> None:
    """Overlay the train label distribution and the predicted distribution on the blind test set."""
    y_train, pred = r["y_train"], r["pred_blind"]
    bins = np.linspace(min(y_train.min(), pred.min()), max(y_train.max(), pred.max()), 31)
    ax.hist(y_train, bins=bins, density=True, alpha=0.45, color="tab:blue", label=f"train labels (n={len(y_train)})")
    ax.hist(pred, bins=bins, density=True, alpha=0.45, color="tab:orange", label=f"test predictions (n={len(pred)})")
    ax.set(title=title, xlabel="pIC50", ylabel="density")
    ax.legend(frameon=False, fontsize=7 if compact else 10)


def draw_heldout_skew(ax, title: str, r: dict, compact: bool = False) -> None:
    """Predicted vs experimental pIC50 on the held-out set, annotated with the mean bias and fitted slope."""
    y_val, pred = r["y_val"], r["pred_val"]
    lo, hi = min(y_val.min(), pred.min()) - 0.2, max(y_val.max(), pred.max()) + 0.2
    bias, slope, intercept = _skew_stats(y_val, pred)
    ax.scatter(y_val, pred, s=10 if compact else 18, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], "k--", label="y = x")
    if np.isfinite(slope):
        xs = np.array([lo, hi])
        ax.plot(xs, slope * xs + intercept, color="tab:red", label=f"fit: slope {slope:.2f}")
    ax.text(
        0.04, 0.96, f"mean bias: {bias:+.3f}\n" + ("predicts high" if bias > 0 else "predicts low"),
        transform=ax.transAxes, va="top", fontsize=7 if compact else 9,
        bbox={"boxstyle": "round", "fc": "w", "alpha": 0.8},
    )
    ax.set(xlim=(lo, hi), ylim=(lo, hi), title=title, xlabel="Experimental pIC50", ylabel="Predicted pIC50")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=False, fontsize=7 if compact else 10)


_DRAW = {"train-test": draw_train_test, "heldout-skew": draw_heldout_skew}


def _select(rows: list[dict]) -> list[dict]:
    """Where a (target, input) group has an ensemble row, keep only the ensemble; otherwise keep each model."""
    ensembled = {(r["target"], r["input"]) for r in rows if r["is_ensemble"]}
    return [r for r in rows if r["is_ensemble"] or (r["target"], r["input"]) not in ensembled]


def _grid(rows: list[dict], kind: str = "") -> tuple[list[str], list[tuple[str, str]], dict]:
    """Isoform columns, (model, input) rows, and the (model, input, isoform) -> row lookup.

    For train-test, rows without blind-set predictions are left out.
    """
    if kind == "train-test":
        rows = [r for r in rows if r.get("pred_blind") is not None]
    isoforms = list(dict.fromkeys(r["target"] for r in rows))
    combos = list(dict.fromkeys((r["name"], r["input"]) for r in rows))
    return isoforms, combos, {(r["name"], r["input"], r["target"]): r for r in rows}


def write_files(kinds: list[str], rows: list[dict], out_dir: Path) -> list[Path]:
    """One PNG per (kind, isoform, model or ensemble, input)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for r in _select(rows):
        isoform = r["target"].split("_")[0]
        for kind in kinds:
            if kind == "train-test" and r.get("pred_blind") is None:
                continue
            path = out_dir / f"{kind}_{_slug(isoform)}_{_slug(r['name'])}_{_slug(r['input'])}.png"
            fig, ax = plt.subplots(figsize=(7, 4.5) if kind == "train-test" else (6, 6))
            _DRAW[kind](ax, f"{isoform}: {r['name']} [{r['input']}]", r)
            fig.tight_layout()
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)
    return written


# --- terminal output -------------------------------------------------------------------------------------------


def image_protocol() -> str | None:
    """The inline-image protocol this terminal is expected to support, or None.

    Environment-based: a terminal can't be asked cheaply and reliably, so an unrecognised terminal, tmux/screen
    (which swallow the escape sequences) or a non-tty stdout all return None and the caller falls back to text.
    """
    if not sys.stdout.isatty() or os.environ.get("TMUX") or os.environ.get("STY"):
        return None
    term, program = os.environ.get("TERM", ""), os.environ.get("TERM_PROGRAM", "")
    if os.environ.get("KITTY_WINDOW_ID") or "kitty" in term or program in ("ghostty", "WezTerm") or "ghostty" in term:
        return "kitty"
    if program == "iTerm.app" or os.environ.get("LC_TERMINAL") == "iTerm2":
        return "iterm2"
    return None


def _montage_png(kind: str, rows: list[dict]) -> bytes:
    """One figure: a column per isoform, a row per (model, input)."""
    isoforms, combos, lookup = _grid(rows, kind)
    fig, axes = plt.subplots(
        len(combos), len(isoforms), figsize=(4.2 * len(isoforms), (3.6 if kind == "train-test" else 4.2) * len(combos)),
        squeeze=False,
    )
    for i, (name, input_label) in enumerate(combos):
        for j, target in enumerate(isoforms):
            ax = axes[i][j]
            r = lookup.get((name, input_label, target))
            if r is None:
                ax.axis("off")
                continue
            _DRAW[kind](ax, f"{target.split('_')[0]}: {name} [{input_label}]", r, compact=True)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def _emit_image(png: bytes, protocol: str) -> None:
    data = base64.standard_b64encode(png).decode()
    out = sys.stdout
    if protocol == "kitty":
        columns = shutil.get_terminal_size().columns
        for k in range(0, len(data), 4096):
            chunk, more = data[k : k + 4096], int(k + 4096 < len(data))
            head = f"a=T,f=100,c={columns},m={more}" if k == 0 else f"m={more}"
            out.write(f"\x1b_G{head};{chunk}\x1b\\")
    else:
        out.write(f"\x1b]1337;File=inline=1;size={len(png)};width=100%;preserveAspectRatio=1:{data}\a")
    out.write("\n")
    out.flush()


def _text_grid(kind: str, rows: list[dict], min_width: int = 44) -> None:
    """Braille/block plots with plotext: isoforms side by side while they fit, else wrapped two or more per row."""
    import plotext as plt_t

    isoforms, combos, lookup = _grid(rows, kind)
    total = shutil.get_terminal_size((120, 40)).columns
    per_row = max(1, min(len(isoforms), total // min_width))
    height = 18
    for name, input_label in combos:
        print(f"\n{kind}: {name} [{input_label}]")
        for start in range(0, len(isoforms), per_row):
            chunk = isoforms[start : start + per_row]
            plt_t.clf()
            plt_t.subplots(1, len(chunk))
            plt_t.plotsize(total, height)
            for j, target in enumerate(chunk):
                r = lookup.get((name, input_label, target))
                plt_t.subplot(1, j + 1)
                plt_t.theme("clear")
                plt_t.title(target.split("_")[0])
                if r is None:
                    continue
                y_val, pred = r["y_val"], r["pred_val"]
                if kind == "train-test":
                    bins = np.linspace(min(r["y_train"].min(), r["pred_blind"].min()), max(r["y_train"].max(), r["pred_blind"].max()), 26)
                    centers = ((bins[:-1] + bins[1:]) / 2).tolist()
                    for label, values, color in (
                        ("train labels", r["y_train"], "blue"), ("test preds", r["pred_blind"], "orange"),
                    ):
                        plt_t.plot(centers, np.histogram(values, bins=bins, density=True)[0].tolist(), label=label, color=color)
                    plt_t.xlabel("pIC50 (density)")
                else:
                    lo, hi = float(min(y_val.min(), pred.min())), float(max(y_val.max(), pred.max()))
                    bias, slope, intercept = _skew_stats(y_val, pred)
                    plt_t.scatter(y_val.tolist(), pred.tolist(), marker="dot", color="blue")
                    plt_t.plot([lo, hi], [lo, hi], color="white")
                    if np.isfinite(slope):
                        plt_t.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color="red")
                    plt_t.xlabel(f"exp  (bias {bias:+.2f}, slope {slope:.2f})")
                    plt_t.ylabel("pred")
            plt_t.show()


def show_terminal(kinds: list[str], rows: list[dict]) -> str:
    """Render a grid per kind in the terminal. Inline images where supported, text plots otherwise.

    Returns what was used: 'kitty', 'iterm2', 'text' or 'none' (with the reason printed to stderr).
    """
    rows = _select(rows)
    protocol = image_protocol()
    if protocol:
        try:
            for kind in kinds:
                print(f"\n{kind}")
                _emit_image(_montage_png(kind, rows), protocol)
            return protocol
        except Exception as exc:  # any failure to draw the image: fall back to text
            print(f"Inline image failed ({exc}); falling back to text plots", file=sys.stderr)
    if not sys.stdout.isatty():
        print("--plots-terminal skipped: stdout is not a terminal", file=sys.stderr)
        return "none"
    try:
        for kind in kinds:
            _text_grid(kind, rows)
    except ImportError:
        print("--plots-terminal needs plotext for text plots: pip install plotext", file=sys.stderr)
        return "none"
    return "text"


# --- from saved results ---------------------------------------------------------------------------------------


def usable_kinds(kinds: list[str], rows: list[dict]) -> list[str]:
    """Drop train-test (with a note) where there are no blind-set predictions to plot."""
    if "train-test" in kinds and not any(r["pred_blind"] is not None for r in rows):
        print("train-test skipped: no blind-set predictions for these results", file=sys.stderr)
        return [k for k in kinds if k != "train-test"]
    return kinds


def rows_from_predictions(
    predictions: Path, blind: Path | None, data_dir: Path, val_split: float, butina_cutoff: float
) -> list[dict]:
    """Rebuild plot rows from an `evaluate-models --predictions-out` CSV (and its `.blind.csv` sibling).

    Train labels are re-derived with load_data from the split and seed recorded in the CSV, so --val-split and
    --butina-cutoff must match the evaluation run. Without the blind file, rows get pred_blind=None and
    train-test can't be drawn.
    """
    from data_tools.load import load_data

    with predictions.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError(f"{predictions} has no rows")
    runs = {(r["split"], int(r["seed"])) for r in records}
    if len(runs) > 1:
        raise ValueError(f"{predictions} mixes several (split, seed) runs {sorted(runs)}; plot one run at a time")
    split, seed = runs.pop()
    train_df, _ = load_data(
        split_type=split, val_fraction=val_split, seed=seed, butina_cutoff=butina_cutoff, data_dir=data_dir,
    )

    blind_preds: dict[tuple[str, str, str], np.ndarray] = {}
    if blind is not None and blind.exists():
        grouped: dict[tuple[str, str, str], list[float]] = {}
        with blind.open(newline="", encoding="utf-8") as stream:
            for r in csv.DictReader(stream):
                grouped.setdefault((r["model"], r["input"], r["isoform"]), []).append(float(r["y_pred"]))
        blind_preds = {k: np.asarray(v) for k, v in grouped.items()}

    val: dict[tuple[str, str, str], tuple[list[float], list[float]]] = {}
    for r in records:
        y, p = val.setdefault((r["model"], r["input"], r["isoform"]), ([], []))
        y.append(float(r["y_true"]))
        p.append(float(r["y_pred"]))

    rows = []
    for (name, input_label, target), (y, p) in val.items():
        labels = train_df[target].to_numpy().astype(float)
        rows.append({
            "target": target, "name": name, "input": input_label, "is_ensemble": name.startswith("ensemble["),
            "y_train": labels[np.isfinite(labels)], "y_val": np.asarray(y), "pred_val": np.asarray(p),
            "pred_blind": blind_preds.get((name, input_label, target)),
        })
    return rows
