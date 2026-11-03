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

import numpy as np
import polars as pl

_SPLIT_COLUMN = "split"
_BLINDLIKE_COLUMN = "split_blindlike"


def _scaffold_of(smiles: str, index: int) -> str:
    """Return the Bemis-Murcko scaffold, or a unique sentinel for unparseable input."""
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles  # type: ignore[import]

    try:
        return MurckoScaffoldSmiles(smiles, includeChirality=False)
    except Exception:  # pylint: disable=broad-except
        return f"__invalid_{index}__"


def _scaffold_groups(smiles: list[str]) -> dict[str, list[int]]:
    """Group row indices by Bemis-Murcko scaffold."""
    groups: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles):
        groups.setdefault(_scaffold_of(smi, idx), []).append(idx)
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


def _morgan_fingerprints(smiles: list[str]) -> list[object | None]:
    """Morgan r2/2048 fingerprints, None where RDKit cannot parse the SMILES."""
    from rdkit.Chem import MolFromSmiles  # type: ignore[import]
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore[import]

    generator = GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints: list[object | None] = []
    for smi in smiles:
        mol = MolFromSmiles(smi)
        fingerprints.append(generator.GetFingerprint(mol) if mol is not None else None)
    return fingerprints


def _nearest_neighbour(query: list[object | None], reference: list[object | None]) -> np.ndarray:
    """Max Tanimoto from each query fingerprint to any reference fingerprint."""
    from rdkit import DataStructs  # type: ignore[import]

    usable = [f for f in reference if f is not None]
    out = np.zeros(len(query), dtype=float)
    if not usable:
        return out
    for i, fingerprint in enumerate(query):
        if fingerprint is None:
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(fingerprint, usable))
    return out


def blindlike_targets(reference_smiles: list[str], pool_smiles: list[str]) -> dict[str, float]:
    """Measure the regime a reference (blind) set occupies relative to a labelled pool.

    Two numbers describe it, and the blind CYP set scores very differently on
    them: its molecules sit *close* to the labelled pool in fingerprint space
    while standing on scaffolds the pool has never seen. A split that matches
    only one of the two misrepresents the submission task.
    """
    pool_scaffolds = {_scaffold_of(smi, i) for i, smi in enumerate(pool_smiles)}
    novel = [
        _scaffold_of(smi, i) not in pool_scaffolds for i, smi in enumerate(reference_smiles)
    ]
    similarity = _nearest_neighbour(
        _morgan_fingerprints(reference_smiles), _morgan_fingerprints(pool_smiles)
    )
    return {
        "unseen_scaffold_fraction": float(np.mean(novel)) if novel else 0.0,
        "median_nn": float(np.median(similarity)) if len(similarity) else 0.0,
        "n_reference": float(len(reference_smiles)),
    }


def _proximity_outside_group(
    fingerprints: list[object | None], group_of: list[int]
) -> np.ndarray:
    """Each molecule's max Tanimoto to pool molecules from a *different* scaffold group.

    Excluding siblings is what makes the number an estimate of the molecule's
    similarity to train once its own scaffold family has been held out.
    """
    from rdkit import DataStructs  # type: ignore[import]

    groups = np.asarray(group_of)
    out = np.zeros(len(fingerprints), dtype=float)
    usable = [i for i, f in enumerate(fingerprints) if f is not None]
    usable_fps = [fingerprints[i] for i in usable]
    usable_groups = groups[usable]
    for position, index in enumerate(usable):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(usable_fps[position], usable_fps))
        sims[usable_groups == groups[index]] = -1.0
        out[index] = max(float(sims.max()), 0.0)
    return out


def _select_units(
    units: list[list[int]],
    proximity: dict[int, np.ndarray],
    target_median: float,
    n_needed: int,
    seed: int,
    candidates_per_step: int = 200,
    truncate: bool = False,
) -> list[list[int]]:
    """Greedily add units, each time picking the one that best centres the median.

    Matching the median rather than taking the units closest to the target keeps
    a realistic spread on either side of it: a held-out slice where every
    molecule sits at exactly 0.57 is not the blind set either.

    A unit is only taken if it fits the remaining quota, so the held-out slice
    comes out the same size whatever regime is being targeted — otherwise two
    configurations are scored on different amounts of data and the comparison
    between them means nothing. `truncate` allows part of a final unit to be
    taken, which is safe only for units that already leave a sibling behind.
    """
    rng = random.Random(seed)
    remaining = list(range(len(units)))
    rng.shuffle(remaining)
    chosen: list[list[int]] = []
    pooled: list[float] = []
    taken = 0

    def cost_of(values: np.ndarray) -> float:
        return abs(float(np.median(pooled + list(values))) - target_median)

    while remaining and taken < n_needed:
        room = n_needed - taken
        fitting = [u for u in remaining[:candidates_per_step] if len(units[u]) <= room]
        if fitting:
            best = min(fitting, key=lambda u: cost_of(proximity[u]))
            chosen.append(list(units[best]))
            pooled.extend(proximity[best])
            taken += len(units[best])
            remaining.remove(best)
            continue
        oversized = [u for u in remaining if len(units[u]) > room]
        if not truncate or not oversized:
            break
        best = min(oversized, key=lambda u: cost_of(proximity[u][:room]))
        chosen.append(list(units[best])[:room])
        pooled.extend(proximity[best][:room])
        taken += room
        remaining.remove(best)
    return chosen


def _slice_statistics(
    held_out: list[int],
    train: list[int],
    fingerprints: dict[int, object | None],
    scaffolds: dict[int, str],
) -> dict[str, float]:
    """Achieved regime of a held-out slice, measured against the rows left in train."""
    if not held_out:
        return {"n": 0.0, "unseen_scaffold_fraction": 0.0, "median_nn": 0.0}
    train_scaffolds = {scaffolds[i] for i in train}
    similarity = _nearest_neighbour(
        [fingerprints[i] for i in held_out], [fingerprints[i] for i in train]
    )
    novel = [scaffolds[i] not in train_scaffolds for i in held_out]
    return {
        "n": float(len(held_out)),
        "unseen_scaffold_fraction": float(np.mean(novel)),
        "median_nn": float(np.median(similarity)),
    }


def assign_blindlike_splits(
    df: pl.DataFrame,
    scored_columns: list[str],
    reference_smiles: list[str] | None = None,
    *,
    targets: dict[str, float] | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[pl.Series, dict]:
    """Return a split Series whose held-out slices reproduce a reference set's regime.

    The pinned scaffold split holds out whole Murcko families, which makes every
    held-out molecule scaffold-novel *and* pushes it far from train. The blind
    CYP set is only the first of those. Selecting which families to hold out, and
    holding part of some families back, matches both axes at once.

    Eligibility follows `assign_splits`: only molecules carrying a scored label
    can enter val or test, and auxiliary-only molecules always go to train.
    """
    present = [c for c in scored_columns if c in df.columns]
    if not present:
        raise ValueError(f"None of the scored columns {scored_columns} are in the table")
    if reference_smiles is None and not targets:
        raise ValueError("Pass reference_smiles to derive targets, or targets directly")

    has_scored = df.select(pl.any_horizontal([pl.col(c).is_not_null() for c in present]))[:, 0]
    pool = [i for i, flag in enumerate(has_scored.to_list()) if flag]
    smiles = df["SMILES"].to_list()
    pool_smiles = [smiles[i] for i in pool]
    print(
        f"{len(pool)} molecules carry >=1 scored label (eligible for val/test); "
        f"{len(df) - len(pool)} auxiliary-only molecules forced to train",
        file=sys.stderr,
    )

    required = ("unseen_scaffold_fraction", "median_nn")
    resolved: dict[str, float] = {}
    if reference_smiles is not None and not (targets and all(k in targets for k in required)):
        measured = blindlike_targets(reference_smiles, pool_smiles)
        resolved = {key: measured[key] for key in required}
        print(
            f"targets from reference (n={int(measured['n_reference'])}): "
            f"unseen scaffolds {resolved['unseen_scaffold_fraction']:.3f}, "
            f"median NN {resolved['median_nn']:.3f}",
            file=sys.stderr,
        )
    for key, value in (targets or {}).items():
        if key in required and resolved.get(key) != value:
            if key in resolved:
                print(f"overriding measured {key}: {resolved[key]:.3f} -> {value:.3f}", file=sys.stderr)
            resolved[key] = value
    absent = [key for key in required if key not in resolved]
    if absent:
        raise ValueError(f"No target for {absent}; pass reference_smiles or give them in targets")
    targets = resolved

    raw_groups = _scaffold_groups(pool_smiles)
    groups = {key: [pool[j] for j in local] for key, local in raw_groups.items()}
    group_index = {key: n for n, key in enumerate(groups)}
    group_of_pool = [0] * len(pool)
    for key, indices in groups.items():
        for i in indices:
            group_of_pool[pool.index(i)] = group_index[key]

    pool_fps = _morgan_fingerprints(pool_smiles)
    fingerprints = {i: pool_fps[n] for n, i in enumerate(pool)}
    scaffolds = {i: _scaffold_of(smiles[i], i) for i in range(len(df))}
    proximity_pool = _proximity_outside_group(pool_fps, group_of_pool)
    proximity = {i: proximity_pool[n] for n, i in enumerate(pool)}

    n_holdout = int(len(pool) * (val_fraction + test_fraction))
    n_unseen = int(round(n_holdout * targets["unseen_scaffold_fraction"]))
    n_seen = n_holdout - n_unseen

    # Whole families: their scaffold leaves train entirely, so every member is novel.
    whole_units = [list(indices) for indices in groups.values()]
    whole_proximity = {
        n: np.array([proximity[i] for i in unit]) for n, unit in enumerate(whole_units)
    }
    chosen_whole = _select_units(
        whole_units, whole_proximity, targets["median_nn"], n_unseen, seed
    )

    # Partial families: one member stays behind, so the scaffold is still in train.
    spoken_for = {i for unit in chosen_whole for i in unit}
    partial_units = [
        sorted(unit)[:-1]
        for unit in whole_units
        if len(unit) >= 2 and not spoken_for.intersection(unit)
    ]
    partial_proximity = {
        n: np.array([proximity[i] for i in unit]) for n, unit in enumerate(partial_units)
    }
    chosen_partial = _select_units(
        partial_units,
        partial_proximity,
        targets["median_nn"],
        n_seen,
        seed + 1,
        truncate=True,
    )

    selected = chosen_whole + chosen_partial
    rng = random.Random(seed)
    rng.shuffle(selected)

    assignment = ["train"] * len(df)
    share = test_fraction / (val_fraction + test_fraction) if val_fraction + test_fraction else 0.5
    n_test_target = int(n_holdout * share)
    n_test = 0
    for unit in selected:
        label = "test" if n_test < n_test_target else "val"
        if label == "test":
            n_test += len(unit)
        for i in unit:
            assignment[i] = label

    held_val = [i for i in range(len(df)) if assignment[i] == "val"]
    held_test = [i for i in range(len(df)) if assignment[i] == "test"]
    train_rows = [i for i in pool if assignment[i] == "train"]
    report = {
        "targets": targets,
        "val": _slice_statistics(held_val, train_rows, fingerprints, scaffolds),
        "test": _slice_statistics(held_test, train_rows, fingerprints, scaffolds),
        "achieved": _slice_statistics(
            held_val + held_test, train_rows, fingerprints, scaffolds
        ),
    }
    print(
        f"blind-matched split (seed {seed}): "
        f"{assignment.count('train')} train / {len(held_val)} val / {len(held_test)} test; "
        f"held out unseen scaffolds {report['achieved']['unseen_scaffold_fraction']:.3f} "
        f"(target {targets['unseen_scaffold_fraction']:.3f}), "
        f"median NN {report['achieved']['median_nn']:.3f} "
        f"(target {targets['median_nn']:.3f})",
        file=sys.stderr,
    )
    return pl.Series(_BLINDLIKE_COLUMN, assignment), report


_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]
