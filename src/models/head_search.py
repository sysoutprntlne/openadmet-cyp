"""Search auxiliary-head subsets for the best transfer onto the four scored isoforms.

The question: auxiliary heads are not free. Each one adds gradient signal, but
a head carrying information already present in another head spends encoder
capacity without adding anything, and a head measuring something only loosely
related to direct inhibition can pull the shared representation away from the
scored task. So the useful auxiliary set is probably a subset, not all of it.

What this does: trains the same model repeatedly, always with the four scored
heads, varying only which auxiliary heads come along, and scores every run on
the identical held-out molecules. The reported number is macro RMSE over the
four scored heads -- auxiliary head accuracy is irrelevant, they exist only to
shape the encoder.

Three things make the comparison valid, and all three matter:

  1. A pinned split column (see data_tools.split_column). Re-splitting per run
     makes split noise the dominant term and the comparison meaningless.
  2. Identical evaluation rows across configurations, which follows from (1).
  3. Repeated seeds. With ~30 configurations and one seed each, the best
     result is very likely the luckiest initialization rather than the best
     subset. Run >=3 seeds and compare means against their spread; treat a
     gap smaller than the seed spread as no difference at all.

Strategies:
  single      baseline plus each candidate alone (cheapest; ranks candidates)
  greedy      forward selection, adding whichever candidate helps most until
              nothing does (finds a good set in O(k^2) runs, not 2^k)
  ablation    all candidates, then all-minus-one (finds the actively harmful)
  exhaustive  every subset (2^k runs; only sane for small k)
"""

import argparse
import itertools
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

_MSE_LINE = re.compile(r"test/(?P<head>[\w.]+)/mse:\s*(?P<value>[-\d.eE+]+)")


def discover_candidates(
    df: pl.DataFrame,
    scored_columns: list[str],
    granularity: str,
) -> dict[str, list[str]]:
    """Group auxiliary label columns into named candidates.

    Grouping is derived from the column name, not a fixed isoform list, so
    heads from any source join the search automatically. Column names are
    `{protein}_{readout}_{source}`, so the leading token is the protein.

    'protein' granularity bundles every auxiliary readout of one protein into
    a single candidate, which directly tests "does this isoform's data help?"
    -- including across sources, so a protein measured by both the qHTS panel
    and ChEMBL is one candidate.
    'source' keeps protein and source separate, which can distinguish a useful
    ChEMBL pIC50 head from a less useful percent-inhibition head on the same
    protein.
    'readout' treats every column separately, the finest and most expensive.
    """
    skip = set(scored_columns) | {"SMILES", "inchikey_block", "inchikey_full", "split", "PUBCHEM_CID"}
    aux = [c for c in df.columns if c not in skip and df[c].dtype.is_numeric()]

    if granularity == "readout":
        return {c: [c] for c in aux}

    candidates: dict[str, list[str]] = {}
    for column in aux:
        parts = column.split("_")
        if granularity == "source" and len(parts) >= 3:
            name = f"{parts[0]}_{parts[-1]}"
        else:
            name = parts[0]
        candidates.setdefault(name, []).append(column)
    return candidates


def _run_once(
    data_path: Path,
    scored_columns: list[str],
    aux_columns: list[str],
    seed: int,
    epochs: int,
    extra_args: list[str],
) -> dict[str, float]:
    """Train one configuration and return {scored_head: test MSE}.

    The model always predicts the scored heads first, so their metrics are
    directly comparable across configurations regardless of what else is
    attached.
    """
    with tempfile.TemporaryDirectory(prefix="head_search_") as tmp:
        cmd = [
            "chemprop",
            "train",
            "--data-path",
            str(data_path),
            "--smiles-columns",
            "SMILES",
            "--target-columns",
            *scored_columns,
            *aux_columns,
            "--task-type",
            "regression",
            "--output-dir",
            tmp,
            "--epochs",
            str(epochs),
            "--pytorch-seed",
            str(seed),
            "--splits-column",
            "split",
            "--show-individual-scores",
            "--remove-checkpoints",
        ] + extra_args
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    output = result.stdout + result.stderr
    if result.returncode != 0:
        tail = "\n".join(line for line in output.splitlines() if line.strip())[-1500:]
        raise RuntimeError(f"chemprop exited {result.returncode}:\n{tail}")

    scores = {m.group("head"): float(m.group("value")) for m in _MSE_LINE.finditer(output)}
    missing = [h for h in scored_columns if h not in scores]
    if missing:
        raise RuntimeError(f"chemprop reported no test score for {missing}; cannot compare configurations")
    return {h: scores[h] for h in scored_columns}


def evaluate_config(
    name: str,
    data_path: Path,
    scored_columns: list[str],
    aux_columns: list[str],
    seeds: list[int],
    epochs: int,
    extra_args: list[str],
) -> dict:
    """Run one configuration across seeds; return macro-RMSE mean/sd and per-head detail."""
    per_seed_macro: list[float] = []
    per_head: dict[str, list[float]] = {h: [] for h in scored_columns}

    for seed in seeds:
        scores = _run_once(data_path, scored_columns, aux_columns, seed, epochs, extra_args)
        rmses = []
        for head, mse in scores.items():
            rmse = math.sqrt(mse)
            per_head[head].append(rmse)
            rmses.append(rmse)
        per_seed_macro.append(sum(rmses) / len(rmses))
        print(f"    seed {seed}: macro RMSE {per_seed_macro[-1]:.4f}", file=sys.stderr)

    mean = statistics.mean(per_seed_macro)
    sd = statistics.stdev(per_seed_macro) if len(per_seed_macro) > 1 else 0.0
    return {
        "name": name,
        "aux_columns": aux_columns,
        "macro_rmse_mean": mean,
        "macro_rmse_sd": sd,
        "per_seed": per_seed_macro,
        "per_head_rmse_mean": {h: statistics.mean(v) for h, v in per_head.items()},
    }


def _plan(strategy: str, candidate_names: list[str]) -> list[tuple[str, list[str]]]:
    """Return the list of (config name, candidate names) to evaluate."""
    if strategy == "single":
        return [("baseline", [])] + [(n, [n]) for n in candidate_names]
    if strategy == "ablation":
        plan = [("baseline", []), ("all", list(candidate_names))]
        return plan + [(f"all-minus-{n}", [m for m in candidate_names if m != n]) for n in candidate_names]
    if strategy == "exhaustive":
        plan = []
        for r in range(len(candidate_names) + 1):
            for combo in itertools.combinations(candidate_names, r):
                plan.append(("baseline" if not combo else "+".join(combo), list(combo)))
        return plan
    return []  # greedy is driven adaptively


def run_search(args, df: pl.DataFrame, candidates: dict[str, list[str]]) -> list[dict]:
    """Execute the chosen strategy and return every evaluated configuration."""
    names = list(candidates)
    seeds = list(range(args.seeds))
    results: list[dict] = []

    def evaluate(name: str, chosen: list[str]) -> dict:
        aux = [c for n in chosen for c in candidates[n]]
        print(f"  [{name}] {len(aux)} auxiliary columns", file=sys.stderr)
        record = evaluate_config(name, args.data_path, args.scored_columns, aux, seeds, args.epochs, args.extra)
        record["candidates"] = chosen
        results.append(record)
        print(f"  [{name}] macro RMSE {record['macro_rmse_mean']:.4f} +/- {record['macro_rmse_sd']:.4f}", file=sys.stderr)
        return record

    if args.strategy == "greedy":
        current: list[str] = []
        best = evaluate("baseline", [])
        remaining = list(names)
        while remaining:
            trials = [evaluate("+".join(current + [n]), current + [n]) for n in remaining]
            winner = min(trials, key=lambda r: r["macro_rmse_mean"])
            gain = best["macro_rmse_mean"] - winner["macro_rmse_mean"]
            tolerance = max(best["macro_rmse_sd"], winner["macro_rmse_sd"])
            if gain <= tolerance:
                print(
                    f"  stopping: best gain {gain:.4f} does not exceed seed spread {tolerance:.4f}",
                    file=sys.stderr,
                )
                break
            added = winner["candidates"][-1]
            current, best = winner["candidates"], winner
            remaining.remove(added)
            print(f"  accepted {added}; set now {current}", file=sys.stderr)
    else:
        for name, chosen in _plan(args.strategy, names):
            evaluate(name, chosen)

    return results


def _print_table(results: list[dict], scored_columns: list[str]) -> None:
    """Print configurations ranked by macro RMSE, best first."""
    ranked = sorted(results, key=lambda r: r["macro_rmse_mean"])
    name_w = max(len(r["name"]) for r in ranked)
    header = f"{'config':{name_w}}  {'macroRMSE':>10}  {'sd':>7}"
    for head in scored_columns:
        header += f"  {head.split('_')[0]:>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in ranked:
        line = f"{r['name']:{name_w}}  {r['macro_rmse_mean']:>10.4f}  {r['macro_rmse_sd']:>7.4f}"
        for head in scored_columns:
            line += f"  {r['per_head_rmse_mean'][head]:>8.4f}"
        print(line)

    best, base = ranked[0], next((r for r in results if r["name"] == "baseline"), None)
    if base is not None and best["name"] != "baseline":
        gain = base["macro_rmse_mean"] - best["macro_rmse_mean"]
        spread = max(best["macro_rmse_sd"], base["macro_rmse_sd"])
        verdict = "exceeds" if gain > spread else "does NOT exceed"
        print(
            f"\nbest={best['name']} beats baseline by {gain:.4f} macro RMSE, which {verdict} "
            f"the seed spread ({spread:.4f})."
        )
        if gain <= spread:
            print("Treat as no measurable difference: add seeds before believing any ranking.")


def main() -> None:
    """Entry point for the head-search CLI."""
    parser = argparse.ArgumentParser(
        description="Find which auxiliary heads actually improve the four scored isoforms, "
        "holding the split and evaluation molecules fixed across configurations.",
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/union_train.csv"))
    parser.add_argument("--scored-columns", nargs="+", default=_SCORED_DEFAULT)
    parser.add_argument(
        "--strategy",
        default="greedy",
        choices=["single", "greedy", "ablation", "exhaustive"],
        help="greedy (default): forward selection. single: rank each candidate alone. "
        "ablation: find harmful heads. exhaustive: every subset (2^k runs)",
    )
    parser.add_argument(
        "--granularity",
        default="protein",
        choices=["protein", "source", "readout"],
        help="Bundle auxiliary columns per protein across sources (default), per protein+source, "
        "or one candidate per column",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per configuration (default: 3)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per run (default: 30)")
    parser.add_argument("--results-path", type=Path, default=Path("results/head_search.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and run count, then exit")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Everything after this flag is passed straight to `chemprop train`",
    )
    args = parser.parse_args()

    if shutil.which("chemprop") is None:
        print("chemprop CLI not found on PATH. Install it with: pip install chemprop", file=sys.stderr)
        sys.exit(1)

    df = pl.read_csv(args.data_path, infer_schema_length=None)
    if "split" not in df.columns:
        print(
            f"{args.data_path} has no 'split' column. It needs a pinned split column: comparing head "
            "subsets across different random splits measures split noise, not head value.",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates = discover_candidates(df, args.scored_columns, args.granularity)
    if not candidates:
        print("No auxiliary label columns found to search over.", file=sys.stderr)
        sys.exit(1)

    print(f"{len(candidates)} candidates ({args.granularity} granularity):", file=sys.stderr)
    for name, cols in candidates.items():
        print(f"  {name}: {len(cols)} column(s) -> {cols}", file=sys.stderr)

    k = len(candidates)
    n_configs = {
        "single": k + 1,
        "ablation": k + 2,
        "exhaustive": 2**k,
        "greedy": 1 + k * (k + 1) // 2,
    }[args.strategy]
    n_runs = n_configs * args.seeds
    print(
        f"\nstrategy={args.strategy}: up to {n_configs} configurations x {args.seeds} seeds = {n_runs} runs "
        f"at {args.epochs} epochs each",
        file=sys.stderr,
    )
    if args.dry_run:
        for name, chosen in _plan(args.strategy, list(candidates)) or [("greedy: adaptive", [])]:
            print(f"  {name}", file=sys.stderr)
        return

    results = run_search(args, df, candidates)
    _print_table(results, args.scored_columns)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
