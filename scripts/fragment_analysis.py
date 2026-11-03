"""Which fragments make the blind_matched validation set look like the blind test set?

Reads results/splits/splits.csv (from compare_splits.py). For each fragment, the
fraction of molecules containing it is measured in the scaffold validation set,
the blind_matched validation set and the blind test set. A fragment "moves toward
the blind set" when blind_matched's rate is closer to the blind test's than the
scaffold split's rate is.

Two fragment vocabularies:
  ring systems  fused/spiro ring clusters, the core chemistry Murcko scaffolds are built from
  BRICS         retrosynthetic leaf and linker pieces, with attachment labels stripped

Outputs in --out-dir: fragments.csv (every fragment with support) and
fragments_report.md / fragments.png (the ones that move).
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import BRICS, Draw


def _ring_systems(mol) -> set[str]:
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    systems: list[set[int]] = []
    for ring in rings:
        merged = [s for s in systems if s & ring]
        for s in merged:
            systems.remove(s)
            ring = ring | s
        systems.append(ring)
    out = set()
    for atoms in systems:
        try:
            out.add(Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), canonical=True))
        except Exception:  # pylint: disable=broad-except
            continue
    return out


def _brics(mol) -> set[str]:
    try:
        pieces = BRICS.BRICSDecompose(mol, minFragmentSize=3, keepNonLeafNodes=False)
    except Exception:  # pylint: disable=broad-except
        return set()
    return {re.sub(r"\[\d+\*\]", "[*]", p) for p in pieces}


def _presence(smiles: list[str], fn) -> list[set[str]]:
    return [fn(Chem.MolFromSmiles(s)) for s in smiles]


def _rates(sets: list[set[str]]) -> Counter:
    c: Counter = Counter()
    for s in sets:
        c.update(s)
    return c


def _table(kind, scaf, bm, blind, min_blind):
    n = {"scaf": len(scaf), "bm": len(bm), "blind": len(blind)}
    cs, cb, ct = _rates(scaf), _rates(bm), _rates(blind)
    rows = []
    for frag in set(cs) | set(cb) | set(ct):
        if ct[frag] + cb[frag] + cs[frag] < min_blind:
            continue
        fs, fb, ft = cs[frag] / n["scaf"], cb[frag] / n["bm"], ct[frag] / n["blind"]
        rows.append(
            {
                "kind": kind, "fragment": frag,
                "n_scaffold_val": cs[frag], "n_blind_matched_val": cb[frag], "n_blind": ct[frag],
                "f_scaffold_val": fs, "f_blind_matched_val": fb, "f_blind": ft,
                # >0: blind_matched's rate is nearer the blind set's than scaffold's is
                "moves_toward_blind": abs(fs - ft) - abs(fb - ft),
                "bm_minus_scaffold": fb - fs,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/splits"))
    ap.add_argument("--min-count", type=int, default=15, help="min molecules containing a fragment across the three sets")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    splits = pl.read_csv(args.out_dir / "splits.csv")
    blind = pl.read_csv(args.data_dir / "test.csv")
    scaf_smi = splits.filter(pl.col("split_scaffold") == "val")["SMILES"].to_list()
    bm_smi = splits.filter(pl.col("split_blind_matched") == "val")["SMILES"].to_list()
    blind_smi = blind["SMILES"].to_list()
    train_smi = splits.filter(pl.col("split_blind_matched") == "train")["SMILES"].to_list()

    frames, vocab = [], {"ring system": _ring_systems, "BRICS": _brics}
    for kind, fn in vocab.items():
        frames.append(_table(kind, _presence(scaf_smi, fn), _presence(bm_smi, fn), _presence(blind_smi, fn), args.min_count))
    table = pl.concat(frames).sort("moves_toward_blind", descending=True)
    table.write_csv(args.out_dir / "fragments.csv")

    lines = [
        "# Fragments: what blind_matched adds that the scaffold validation set lacks",
        "",
        f"Validation sets: scaffold {len(scaf_smi)}, blind_matched {len(bm_smi)}; blind test {len(blind_smi)} molecules. "
        "Rates are the % of molecules containing the fragment. Fragments need >= "
        f"{args.min_count} molecules across the three sets.",
        "",
        "`gain` = |scaffold - blind| - |blind_matched - blind| in percentage points: positive means "
        "blind_matched's rate is closer to the blind set's.",
        "",
    ]
    fig_items = []
    for kind in vocab:
        sub = table.filter(pl.col("kind") == kind)
        closer = sub.filter(pl.col("moves_toward_blind") > 0).head(args.top)
        further = sub.sort("moves_toward_blind").head(args.top)
        both = sub.filter((pl.col("f_blind") > pl.col("f_scaffold_val")) & (pl.col("f_blind_matched_val") > pl.col("f_scaffold_val"))).sort(
            "bm_minus_scaffold", descending=True
        ).head(args.top)
        for title, df in (
            (f"{kind}: moves toward the blind set", closer),
            (f"{kind}: moves away from the blind set", further),
        ):
            lines += [f"## {title}", "", "| fragment | scaffold val | blind_matched val | blind | gain (pp) |", "|---|---|---|---|---|"]
            for r in df.iter_rows(named=True):
                lines.append(
                    f"| `{r['fragment']}` | {r['f_scaffold_val']:.1%} | {r['f_blind_matched_val']:.1%} | {r['f_blind']:.1%} | {100 * r['moves_toward_blind']:+.1f} |"
                )
            lines.append("")
        if kind == "ring system":
            fig_items = closer.head(12)

    # Ring systems each set carries that its own train set lacks (blind: all of train.csv).
    train_of = {
        "scaffold val": splits.filter(pl.col("split_scaffold") == "train")["SMILES"].to_list(),
        "blind_matched val": train_smi,
        "blind test": splits["SMILES"].to_list(),
    }
    lines += ["## Molecules carrying a ring system absent from their own train set", ""]
    for name, smi in (("scaffold val", scaf_smi), ("blind_matched val", bm_smi), ("blind test", blind_smi)):
        seen = _rates(_presence(train_of[name], _ring_systems))
        sets = _presence(smi, _ring_systems)
        lines.append(f"- {name}: {np.mean([any(r not in seen for r in s) for s in sets]):.1%}")
    lines.append("")

    (args.out_dir / "fragments_report.md").write_text("\n".join(lines))

    if len(fig_items):
        mols = [Chem.MolFromSmiles(f) or Chem.MolFromSmarts(f) for f in fig_items["fragment"].to_list()]
        legends = [
            f"scaf {r['f_scaffold_val']:.1%} | bm {r['f_blind_matched_val']:.1%} | blind {r['f_blind']:.1%}"
            for r in fig_items.iter_rows(named=True)
        ]
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 240), legends=legends)
        img.save(args.out_dir / "fragments.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
