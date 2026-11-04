# tests/test_split_column.py
"""Blind-matched split construction: eligibility rules and the two targets it matches."""

import numpy as np
import polars as pl
import pytest

from data_tools.split_column import (
    assign_blindlike_splits,
    blindlike_targets,
)

SCORED = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

# Ten distinct ring systems, so Murcko grouping produces ten scaffold families,
# each with six substituent variants that share the family's scaffold.
_CORES = [
    "{r}c1ccccc1",
    "{r}c1ccncc1",
    "{r}c1ccc2ccccc2c1",
    "{r}c1ccc2[nH]cnc2c1",
    "{r}c1cccs1",
    "{r}c1ccc(-c2ccccc2)cc1",
    "{r}c1ccc2ncccc2c1",
    "{r}C1CCNCC1",
    "{r}c1ccc(N2CCOCC2)cc1",
    "{r}c1cnc2ccccc2c1",
]
_SUBSTITUENTS = ["C", "CC", "CCC", "CO", "Cl", "N#C"]


def _library() -> tuple[list[str], list[str]]:
    names, smiles = [], []
    for core_index, core in enumerate(_CORES):
        for sub_index, sub in enumerate(_SUBSTITUENTS):
            names.append(f"OCNT-{core_index}{sub_index}")
            smiles.append(core.format(r=sub))
    return names, smiles


@pytest.fixture()
def table() -> pl.DataFrame:
    """Sixty scored molecules across ten scaffold families, plus four auxiliary-only rows."""
    names, smiles = _library()
    n = len(smiles)
    frame = pl.DataFrame(
        {
            "Molecule_Name": names,
            "SMILES": smiles,
            **{column: [5.0 + (i % 7) * 0.1 for i in range(n)] for column in SCORED},
        }
    )
    auxiliary = pl.DataFrame(
        {
            "Molecule_Name": ["AUX-1", "AUX-2", "AUX-3", "AUX-4"],
            "SMILES": ["CCO", "CCCCO", "CC(=O)O", "CCN"],
            **{column: [None] * 4 for column in SCORED},
        },
        schema=frame.schema,
    )
    return frame.vstack(auxiliary)


def _holdout_smiles(table: pl.DataFrame, splits: pl.Series) -> list[str]:
    return table.filter(splits.is_in(["val", "test"]))["SMILES"].to_list()


def test_auxiliary_only_rows_are_forced_to_train(table):
    splits, _ = assign_blindlike_splits(
        table, SCORED, targets={"unseen_scaffold_fraction": 0.75, "median_nn": 0.5}, seed=0
    )
    auxiliary = table.filter(pl.col("Molecule_Name").str.starts_with("AUX-"))
    assigned = splits.filter(table["Molecule_Name"].str.starts_with("AUX-"))
    assert len(auxiliary) == 4
    assert set(assigned.to_list()) == {"train"}


def test_every_split_label_is_used(table):
    splits, _ = assign_blindlike_splits(
        table, SCORED, targets={"unseen_scaffold_fraction": 0.75, "median_nn": 0.5}, seed=0
    )
    assert set(splits.to_list()) == {"train", "val", "test"}


def test_unseen_scaffold_fraction_tracks_the_target(table):
    """The held-out slice should mix novel scaffolds with scaffolds train still carries."""
    mostly_novel, novel_report = assign_blindlike_splits(
        table, SCORED, targets={"unseen_scaffold_fraction": 1.0, "median_nn": 0.5}, seed=0
    )
    mixed, mixed_report = assign_blindlike_splits(
        table, SCORED, targets={"unseen_scaffold_fraction": 0.5, "median_nn": 0.5}, seed=0
    )
    assert novel_report["achieved"]["unseen_scaffold_fraction"] == pytest.approx(1.0)
    assert mixed_report["achieved"]["unseen_scaffold_fraction"] < 0.9
    assert len(_holdout_smiles(table, mostly_novel)) == len(_holdout_smiles(table, mixed))


def test_a_seen_scaffold_leaves_a_sibling_in_train(table):
    """A held-out molecule only counts as 'seen' if its scaffold survives in train."""
    from data_tools.split_column import _scaffold_of

    splits, report = assign_blindlike_splits(
        table, SCORED, targets={"unseen_scaffold_fraction": 0.5, "median_nn": 0.5}, seed=0
    )
    train_scaffolds = {
        _scaffold_of(smiles, index)
        for index, smiles in enumerate(table.filter(splits == "train")["SMILES"].to_list())
    }
    held_out = [
        _scaffold_of(smiles, index) for index, smiles in enumerate(_holdout_smiles(table, splits))
    ]
    seen = sum(scaffold in train_scaffolds for scaffold in held_out)
    assert seen > 0
    assert seen / len(held_out) == pytest.approx(
        1.0 - report["achieved"]["unseen_scaffold_fraction"], abs=1e-9
    )


def test_median_similarity_target_is_honoured(table):
    """Asking for a near neighbourhood must not return the same slice as asking for a far one."""
    near, near_report = assign_blindlike_splits(
        table, SCORED, targets={"unseen_scaffold_fraction": 0.75, "median_nn": 0.95}, seed=0
    )
    far, far_report = assign_blindlike_splits(
        table, SCORED, targets={"unseen_scaffold_fraction": 0.75, "median_nn": 0.05}, seed=0
    )
    assert near_report["achieved"]["median_nn"] > far_report["achieved"]["median_nn"]
    assert _holdout_smiles(table, near) != _holdout_smiles(table, far)


def test_report_states_what_was_asked_and_what_was_achieved(table):
    targets = {"unseen_scaffold_fraction": 0.75, "median_nn": 0.5}
    _, report = assign_blindlike_splits(table, SCORED, targets=targets, seed=0)
    assert report["targets"] == targets
    for slice_name in ("val", "test"):
        assert report[slice_name]["n"] > 0
        assert 0.0 <= report[slice_name]["median_nn"] <= 1.0
    assert report["achieved"]["n"] == report["val"]["n"] + report["test"]["n"]


def test_targets_are_derived_from_a_reference_set(table):
    """Targets come from the blind set itself rather than being hardcoded."""
    _, reference = _library()
    pool = table.filter(pl.col("Molecule_Name").str.starts_with("OCNT-"))["SMILES"].to_list()
    targets = blindlike_targets(reference[:12], pool)
    assert targets["n_reference"] == 12
    # Every reference molecule is drawn from the pool, so no scaffold is novel
    # and each one finds itself as its own nearest neighbour.
    assert targets["unseen_scaffold_fraction"] == pytest.approx(0.0)
    assert targets["median_nn"] == pytest.approx(1.0)


def test_splits_are_deterministic_for_a_seed(table):
    targets = {"unseen_scaffold_fraction": 0.75, "median_nn": 0.5}
    first, _ = assign_blindlike_splits(table, SCORED, targets=targets, seed=7)
    second, _ = assign_blindlike_splits(table, SCORED, targets=targets, seed=7)
    assert first.to_list() == second.to_list()


def test_fractions_size_the_held_out_slices(table):
    splits, report = assign_blindlike_splits(
        table,
        SCORED,
        targets={"unseen_scaffold_fraction": 0.75, "median_nn": 0.5},
        val_fraction=0.2,
        test_fraction=0.2,
        seed=0,
    )
    n_eligible = 60
    assert report["achieved"]["n"] >= int(n_eligible * 0.4) * 0.8
    assert np.isclose(len(splits), len(table))
