"""Validate ensemble input data, generate molecule identifiers, and create
group-aware OOF folds in the pinned training split.

Use full inchikeys for row identity and connectivity blocks for grouping.
Preserve existing train/validation/test assignments.
"""

import random
from typing import Literal

import polars as pl

from data_tools.split_column import _butina_groups, _scaffold_groups
from data_tools.standardize import connectivity_keys, stereo_keys

SCORED_TARGETS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

def prepare_ensemble_data(df: pl.DataFrame) -> pl.DataFrame:
    """Validate prepared data and add verified molecule identifiers."""
    
    missing = {"SMILES", "split", *SCORED_TARGETS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.is_empty():
        raise ValueError("Data is empty")

    smiles = df["SMILES"]
    if smiles.dtype != pl.String or smiles.null_count():
        raise ValueError("SMILES must be non-null strings")
    if (smiles.str.strip_chars() == "").any():
        raise ValueError("SMILES cannot be blank")

    for target in SCORED_TARGETS:
        values = df[target]
        if values.dtype != pl.Null and not values.dtype.is_numeric():
            raise ValueError(f"{target} must be numeric")
        if values.cast(pl.Float64).is_infinite().any():
            raise ValueError(f"{target} contains infinity")

    splits = df["split"]
    if splits.null_count() or not set(splits.to_list()) <= {"train", "val", "test"}:
        raise ValueError("split must contain only train, val, or test")

    generated = {
        "inchikey_block": connectivity_keys(smiles.to_list()),
        "inchikey_full": stereo_keys(smiles.to_list()),
    }
    for name, values in generated.items():
        if any(value is None for value in values):
            raise ValueError(f"Could not generate {name} for every row")
  
        if name in df.columns and df[name].to_list() != values:
            raise ValueError(f"Existing {name} does not match SMILES")
        df = df.with_columns(pl.Series(name, values, dtype=pl.String))

    if df["inchikey_full"].n_unique() != df.height:
        raise ValueError("Duplicate full InChIKeys")

    split_counts = df.group_by("inchikey_block").agg(
        pl.col("split").n_unique().alias("n_splits")
    )
    if (split_counts["n_splits"] > 1).any():
        raise ValueError("A connectivity block appears in multiple splits")

    return df


def make_oof_folds(
    df: pl.DataFrame,
    *,
    strategy: Literal["scaffold", "butina", "random"] = "scaffold",
    n_folds: int = 5, # consider making it 10
    seed: int = 0, 
    butina_cutoff: float = 0.4,

) -> pl.DataFrame:
    """Expects prepared data. Create OOF folds within pinned training data.

    Keep scaffold/Butina groups or randomized connectivity blocks together.
    Require usable labels for all four scored targets in each fold's
    training and held-out populations.

    Return inchikey_full, fold, and oof_group . The caller saves the assignments and
    settings (strategy, n_folds, seed, and Butina cutoff when applicable).
    """

    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    
    train = df.filter(pl.col("split") == "train")
    if train.is_empty():
        raise ValueError("No pinned training rows")
    smiles = train["SMILES"].to_list()

    if strategy == "scaffold":
        groups = _scaffold_groups(smiles)
    elif strategy == "butina":
        groups = _butina_groups(smiles, butina_cutoff)
    elif strategy == "random":
        groups = {}
        for row_index, block in enumerate(train["inchikey_block"]):
            groups.setdefault(block, []).append(row_index)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if len(groups) < n_folds:
        raise ValueError("Not enough groups for the requested folds")

    ordered_groups = list(groups.values())
    random.Random(seed).shuffle(ordered_groups)

    ordered_groups.sort(key=len, reverse=True)

    fold_sizes = [0] * n_folds
    assignments = [-1] * train.height

    for rows in ordered_groups:
        fold = min(range(n_folds), key=lambda i: fold_sizes[i])

        for row_index in rows:
            assignments[row_index] = fold

        fold_sizes[fold] += len(rows)

    group_ids = [""] * train.height
    for group_number, rows in enumerate(groups.values()):
        for row_index in rows:
            group_ids[row_index] = f"group_{group_number}"

    assigned = train.with_columns(
        pl.Series("fold", assignments),
        pl.Series("oof_group", group_ids),
    )

    if assigned["inchikey_full"].n_unique() != assigned.height:
        raise ValueError("Duplicate full inchikeys in fold assignments")

    block_folds = assigned.group_by("inchikey_block").agg(
        pl.col("fold").n_unique().alias("n_folds")
    )
    if (block_folds["n_folds"] > 1).any():
        raise ValueError("A connectivity block appears in multiple folds")

    if -1 in assignments or any(size == 0 for size in fold_sizes):
        raise ValueError("Every row must be assigned and every fold nonempty")

    for fold in range(n_folds):
        for name, population in [
            ("held-out", assigned.filter(pl.col("fold") == fold)),
            ("training", assigned.filter(pl.col("fold") != fold)),
        ]:
            for target in SCORED_TARGETS:
                labels = population[target].cast(pl.Float64)
                observed = (
                    labels.is_not_null() & labels.is_finite()
                ).fill_null(False)

                if not observed.any():
                    raise ValueError(
                        f"Fold {fold}: no usable {name} labels for {target}"
                    )

    return assigned.select("inchikey_full", "fold", "oof_group")
        
  