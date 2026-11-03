"""Save the scaffold and blind_matched splits and compare their validation chemical space.

Splits come from `data_tools.load.load_data`, with the same defaults as
`evaluate-models` (val fraction 0.2, seed 42), so the saved files describe exactly
what that CLI evaluates on.

Outputs (default results/splits/):
  splits.csv          Molecule_Name, SMILES, split_scaffold, split_blind_matched
  summary.csv         one row per slice (scaffold val, blind_matched val, blind test)
  properties.csv      bulk property mean/sd per slice
  heads.csv           labelled-molecule counts and pIC50 mean/sd per head per slice
  chemical_space.png  PCA of Morgan fingerprints and nearest-neighbour similarity
  report.md           the written comparison
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from data_tools.load import load_data, load_test, load_train
from data_tools.split_column import _scaffold_of

HEADS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]
BASIC_N = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N#*);!$(N=*)]")


def _fps(smiles):
    gen = GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def _nn(query, reference, exclude_self=False):
    out = np.zeros(len(query))
    for i, fp in enumerate(query):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, reference))
        if exclude_self:
            sims[i] = -1.0
        out[i] = sims.max()
    return out


def _props(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(
            {
                "MW": Descriptors.MolWt(m),
                "cLogP": Descriptors.MolLogP(m),
                "TPSA": rdMolDescriptors.CalcTPSA(m),
                "HBD": Lipinski.NumHDonors(m),
                "RotB": Lipinski.NumRotatableBonds(m),
                "AromRings": rdMolDescriptors.CalcNumAromaticRings(m),
                "HeavyAtoms": m.GetNumHeavyAtoms(),
                "BasicN": float(m.HasSubstructMatch(BASIC_N)),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/splits"))
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = load_train(args.data_dir / "train.csv")
    blind = load_test(args.data_dir / "test.csv")
    n = len(full)

    # Map each split's val rows back to row positions in train.csv (SMILES is the key).
    row_of = {}
    for i, s in enumerate(full["SMILES"].to_list()):
        row_of.setdefault(s, []).append(i)

    def val_mask(strategy):
        train_df, val_df = load_data(
            split_type=strategy, val_fraction=args.val_split, seed=args.seed, data_dir=args.data_dir
        )
        mask = np.zeros(n, dtype=bool)
        pending = {k: list(v) for k, v in row_of.items()}
        for s in val_df["SMILES"].to_list():
            mask[pending[s].pop()] = True
        assert mask.sum() == len(val_df) and n - mask.sum() == len(train_df)
        return mask

    masks = {"scaffold": val_mask("scaffold"), "blind_matched": val_mask("blind_matched")}
    pl.DataFrame(
        {
            "Molecule_Name": full["Molecule_Name"],
            "SMILES": full["SMILES"],
            "split_scaffold": np.where(masks["scaffold"], "val", "train"),
            "split_blind_matched": np.where(masks["blind_matched"], "val", "train"),
        }
    ).write_csv(args.out_dir / "splits.csv")

    smiles = np.array(full["SMILES"].to_list())
    fps = _fps(smiles)
    blind_smiles = blind["SMILES"].to_list()
    blind_fps = _fps(blind_smiles)
    scaffolds = np.array([_scaffold_of(s, i) for i, s in enumerate(smiles)])
    blind_scaffolds = [_scaffold_of(s, i) for i, s in enumerate(blind_smiles)]
    props = _props(smiles)
    blind_props = _props(blind_smiles)

    slices, summary, prop_rows, head_rows, nn_to_train = {}, [], [], [], {}
    for name, mask in masks.items():
        train_idx, val_idx = np.where(~mask)[0], np.where(mask)[0]
        train_fps = [fps[i] for i in train_idx]
        val_fps = [fps[i] for i in val_idx]
        train_scaf = set(scaffolds[train_idx])
        nn = _nn(val_fps, train_fps)
        nn_to_train[name] = nn
        # similarity of val to the blind set, and of the blind set to this split's train
        summary.append(
            {
                "slice": f"{name} val",
                "n": len(val_idx),
                "novel_scaffold_frac": float(np.mean([s not in train_scaf for s in scaffolds[val_idx]])),
                "nn_train_p10": float(np.quantile(nn, 0.1)),
                "nn_train_median": float(np.median(nn)),
                "nn_train_p90": float(np.quantile(nn, 0.9)),
                "frac_nn_below_0.4": float(np.mean(nn < 0.4)),
                "nn_blind_median": float(np.median(_nn(val_fps, blind_fps))),
                "n_scaffolds": len(set(scaffolds[val_idx])),
            }
        )
        slices[name] = (train_idx, val_idx)
        sub = props[val_idx.tolist()]
        for c in props.columns:
            prop_rows.append({"slice": f"{name} val", "property": c, "mean": sub[c].mean(), "sd": sub[c].std()})
        for h in HEADS:
            col = full[h].to_numpy()[val_idx]
            col = col[~np.isnan(col)]
            head_rows.append(
                {"slice": f"{name} val", "head": h.split("_")[0], "n_labelled": len(col),
                 "mean": float(col.mean()), "sd": float(col.std())}
            )

    # Reference rows: the blind set measured against the full labelled table, and the train table itself.
    train_all_scaf = set(scaffolds)
    nn_blind = _nn(blind_fps, fps)
    summary.append(
        {
            "slice": "blind test (vs all train.csv)",
            "n": len(blind),
            "novel_scaffold_frac": float(np.mean([s not in train_all_scaf for s in blind_scaffolds])),
            "nn_train_p10": float(np.quantile(nn_blind, 0.1)),
            "nn_train_median": float(np.median(nn_blind)),
            "nn_train_p90": float(np.quantile(nn_blind, 0.9)),
            "frac_nn_below_0.4": float(np.mean(nn_blind < 0.4)),
            "nn_blind_median": float("nan"),
            "n_scaffolds": len(set(blind_scaffolds)),
        }
    )
    for c in props.columns:
        prop_rows.append({"slice": "blind test", "property": c, "mean": blind_props[c].mean(), "sd": blind_props[c].std()})
        prop_rows.append({"slice": "all train.csv", "property": c, "mean": props[c].mean(), "sd": props[c].std()})

    # Overlap of the two validation sets.
    a, b = masks["scaffold"], masks["blind_matched"]
    overlap = int((a & b).sum())
    jaccard = overlap / int((a | b).sum())
    scaf_overlap = len(set(scaffolds[a]) & set(scaffolds[b]))

    # Distance between the validation sets in property space (standardised by train.csv sd).
    def standardised_shift(mask):
        sub = props[np.where(mask)[0].tolist()]
        return {c: (sub[c].mean() - props[c].mean()) / props[c].std() for c in props.columns}

    shift = {k: standardised_shift(m) for k, m in masks.items()}
    blind_shift = {c: (blind_props[c].mean() - props[c].mean()) / props[c].std() for c in props.columns}

    summary_df, prop_df, head_df = pl.DataFrame(summary), pl.DataFrame(prop_rows), pl.DataFrame(head_rows)
    summary_df.write_csv(args.out_dir / "summary.csv")
    prop_df.write_csv(args.out_dir / "properties.csv")
    head_df.write_csv(args.out_dir / "heads.csv")

    # --- figure ---
    from sklearn.decomposition import PCA

    def bits(fp_list):
        out = np.zeros((len(fp_list), 2048), dtype=np.uint8)
        for i, fp in enumerate(fp_list):
            DataStructs.ConvertToNumpyArray(fp, out[i])
        return out

    pca = PCA(n_components=2, random_state=0).fit(bits(fps))
    xy, bxy = pca.transform(bits(fps)), pca.transform(bits(blind_fps))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for k, (name, color) in enumerate([("scaffold", "tab:red"), ("blind_matched", "tab:blue")]):
        ax[k].scatter(xy[:, 0], xy[:, 1], s=3, c="lightgrey", label="train.csv")
        ax[k].scatter(bxy[:, 0], bxy[:, 1], s=6, c="black", alpha=0.5, label="blind test")
        v = slices[name][1]
        ax[k].scatter(xy[v, 0], xy[v, 1], s=6, c=color, alpha=0.7, label=f"{name} val")
        ax[k].set_title(f"{name} val vs blind test (Morgan PCA)")
        ax[k].legend(markerscale=3, fontsize=8)
    bins = np.linspace(0.2, 1.0, 33)
    for name, color in [("scaffold", "tab:red"), ("blind_matched", "tab:blue")]:
        ax[2].hist(nn_to_train[name], bins=bins, histtype="step", color=color, density=True, label=f"{name} val -> its train")
    ax[2].hist(nn_blind, bins=bins, histtype="step", color="black", density=True, label="blind -> train.csv")
    ax[2].set_xlabel("max Tanimoto to train")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "chemical_space.png", dpi=150)

    # --- report ---
    def row(slice_name):
        return summary_df.filter(pl.col("slice") == slice_name).row(0, named=True)

    s, m, bl = row("scaffold val"), row("blind_matched val"), row("blind test (vs all train.csv)")
    lines = [
        "# Scaffold vs blind_matched validation sets",
        "",
        f"Seed {args.seed}, val fraction {args.val_split}, {n} molecules in train.csv. "
        "Splits are in `splits.csv` (`split_scaffold`, `split_blind_matched`).",
        "",
        "## Chemical space",
        "",
        "| slice | n | novel scaffolds | NN to train p10 / median / p90 | NN < 0.4 | distinct scaffolds |",
        "|---|---|---|---|---|---|",
    ]
    for r in (s, m, bl):
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['novel_scaffold_frac']:.1%} | "
            f"{r['nn_train_p10']:.3f} / {r['nn_train_median']:.3f} / {r['nn_train_p90']:.3f} | "
            f"{r['frac_nn_below_0.4']:.1%} | {r['n_scaffolds']} |"
        )
    lines += [
        "",
        f"Median similarity of each validation set to the blind set: scaffold {s['nn_blind_median']:.3f}, "
        f"blind_matched {m['nn_blind_median']:.3f}.",
        "",
        "## How the two validation sets differ",
        "",
        f"- Overlap: {overlap} molecules in both ({jaccard:.1%} Jaccard); {scaf_overlap} scaffolds appear in both.",
        "",
        "Bulk property shift, in sd of train.csv (positive = higher than train):",
        "",
        "| property | scaffold val | blind_matched val | blind test |",
        "|---|---|---|---|",
    ]
    for c in props.columns:
        lines.append(f"| {c} | {shift['scaffold'][c]:+.2f} | {shift['blind_matched'][c]:+.2f} | {blind_shift[c]:+.2f} |")
    lines += ["", "Labelled molecules and pIC50 per head in each validation set:", "", "| head | scaffold n / mean | blind_matched n / mean |", "|---|---|---|"]
    for h in HEADS:
        hn = h.split("_")[0]
        a_ = head_df.filter((pl.col("slice") == "scaffold val") & (pl.col("head") == hn)).row(0, named=True)
        b_ = head_df.filter((pl.col("slice") == "blind_matched val") & (pl.col("head") == hn)).row(0, named=True)
        lines.append(f"| {hn} | {a_['n_labelled']} / {a_['mean']:.2f} | {b_['n_labelled']} / {b_['mean']:.2f} |")
    lines += ["", "![chemical space](chemical_space.png)", ""]
    (args.out_dir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
