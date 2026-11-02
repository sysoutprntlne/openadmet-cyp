"""Write a fixed train/val/test split column into a multi-task table.

Why a *pinned* split matters for head-subset experiments: if every training
run re-splits at random, the run-to-run difference caused by the split is
comparable to (or larger than) the effect of adding an auxiliary head. Any
comparison between head subsets is then measuring split noise. Pinning one
split column and reusing it across every configuration removes that term
entirely, so a difference between subsets is attributable to the heads.

Two rules shape the split:

  1. Only molecules carrying at least one *scored* label can land in val or
     test. Evaluation must happen on the challenge distribution; a metric
     averaged over auxiliary-only molecules answers a different question.
  2. Auxiliary-only molecules always go to train. They have no scored label,
     so holding them out buys nothing and costs training signal.

Grouping is by scaffold (default) or Butina cluster, so a scaffold family
never straddles the split, which would leak structure between train and test.
"""

import random
import sys
from typing import Literal

import polars as pl

_SPLIT_COLUMN = "split"


def _scaffold_groups(smiles: list[str]) -> dict[str, list[int]]:
    """Group row indices by Bemis-Murcko scaffold."""
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles  # type: ignore[import]

    groups: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles):
        try:
            scaffold = MurckoScaffoldSmiles(smi, includeChirality=False)
        except Exception:  # pylint: disable=broad-except
            scaffold = f"__invalid_{idx}__"
        groups.setdefault(scaffold, []).append(idx)
    return groups


def _butina_groups(smiles: list[str], cutoff: float) -> dict[str, list[int]]:
    """Group row indices by Butina cluster on Morgan fingerprints."""
    from rdkit import DataStructs  # type: ignore[import]
    from rdkit.Chem import MolFromSmiles  # type: ignore[import]
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore[import]
    from rdkit.ML.Cluster import Butina  # type: ignore[import]

    morgan = GetMorganGenerator(radius=2, fpSize=2048)
    fps, valid_indices, invalid = [], [], []
    for idx, smi in enumerate(smiles):
        mol = MolFromSmiles(smi)
        if mol is None:
            invalid.append(idx)
            continue
        fps.append(morgan.GetFingerprint(mol))
        valid_indices.append(idx)

    dists: list[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    raw = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    groups = {f"cluster_{i}": [valid_indices[j] for j in cluster] for i, cluster in enumerate(raw)}
    for idx in invalid:
        groups[f"__invalid_{idx}__"] = [idx]
    return groups


def assign_splits(
    df: pl.DataFrame,
    scored_columns: list[str],
    strategy: Literal["scaffold", "butina", "random"] = "scaffold",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
    butina_cutoff: float = 0.4,
) -> pl.Series:
    """Return a split Series ('train'/'val'/'test') aligned to df's rows.

    Auxiliary-only rows (no non-null scored column) are forced to 'train'.
    Grouped strategies keep a scaffold or cluster family within one split.
    """
    present = [c for c in scored_columns if c in df.columns]
    if not present:
        raise ValueError(f"None of the scored columns {scored_columns} are in the table")

    has_scored = df.select(pl.any_horizontal([pl.col(c).is_not_null() for c in present]))[:, 0].to_list()
    scored_rows = [i for i, flag in enumerate(has_scored) if flag]
    print(
        f"{len(scored_rows)} molecules carry >=1 scored label (eligible for val/test); "
        f"{len(df) - len(scored_rows)} auxiliary-only molecules forced to train",
        file=sys.stderr,
    )

    smiles = df["SMILES"].to_list()
    if strategy == "random":
        groups = {str(i): [i] for i in scored_rows}
    else:
        sub_smiles = [smiles[i] for i in scored_rows]
        raw_groups = _scaffold_groups(sub_smiles) if strategy == "scaffold" else _butina_groups(sub_smiles, butina_cutoff)
        # Remap local indices back onto df row indices.
        groups = {key: [scored_rows[j] for j in local] for key, local in raw_groups.items()}

    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    random.Random(seed).shuffle(ordered)

    n_eligible = len(scored_rows)
    n_test_target = int(n_eligible * test_fraction)
    n_val_target = int(n_eligible * val_fraction)

    assignment = ["train"] * len(df)
    n_test = n_val = 0
    for _key, indices in ordered:
        if n_test < n_test_target:
            label, n_test = "test", n_test + len(indices)
        elif n_val < n_val_target:
            label, n_val = "val", n_val + len(indices)
        else:
            label = "train"
        for i in indices:
            assignment[i] = label

    print(
        f"split ({strategy}, seed {seed}): "
        f"{assignment.count('train')} train / {assignment.count('val')} val / {assignment.count('test')} test",
        file=sys.stderr,
    )
    return pl.Series(_SPLIT_COLUMN, assignment)


_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]
