# tests/test_merge_datasets.py
"""Run-time merge of selected datasets into per-source heads."""

import polars as pl
import pytest

from data_tools.merge_datasets import merge, parse_dataset_specs

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
IBUPROFEN = "CC(C)CC1=CC=C(C=C1)C(=O)O"
PHENOL = "C1=CC=C(C=C1)O"
TOLUENE = "CC1=CC=CC=C1"


@pytest.fixture()
def data_dir(tmp_path):
    names = ["OCNT-1", "OCNT-2", "OCNT-3"]
    smiles = [ASPIRIN, CAFFEINE, IBUPROFEN]
    pl.DataFrame(
        {
            "Molecule_Name": names,
            "SMILES": smiles,
            "CYP2D6_pIC50_direct_inhibition": [5.0, None, 6.0],
            "CYP3A4_pIC50_direct_inhibition": [4.5, 5.5, None],
        }
    ).write_csv(tmp_path / "train.csv")
    pl.DataFrame(
        {
            "Molecule_Name": names + ["OCNT-4"],
            "SMILES": smiles + [PHENOL],
            "CYP3A4_pIC50_TDI_condition": [4.0, None, 5.0, 6.0],
        }
    ).write_csv(tmp_path / "tdi_train.csv")
    pl.DataFrame(
        {
            "SMILES": [ASPIRIN, TOLUENE, "CCO"],
            "CYP2D6_pIC50_bindingdb": [7.0, 5.0, None],
            "CYP2D6_pIC50_chembl": [None, None, 3.0],
            "CYP1A2_pIC50_bindingdb": [6.0, None, None],
        }
    ).write_csv(tmp_path / "family_heads.csv")
    return tmp_path


def _merge(data_dir, specs):
    return merge(parse_dataset_specs(specs), data_dir=data_dir, test_path=None, split_strategy="random")


def test_challenge_only_has_scored_heads(data_dir):
    table, heads, scored, _ = _merge(data_dir, [])
    assert heads == scored == ["CYP2D6-Direct", "CYP3A4-Direct"]
    assert len(table) == 3


def test_heads_are_split_by_source(data_dir):
    table, heads, _, _ = _merge(data_dir, ["tdi", "bindingdb"])
    assert {"CYP3A4-TDI", "CYP2D6-BindingDB", "CYP1A2-BindingDB"} <= set(heads)
    aspirin = table.filter(pl.col("SMILES") == ASPIRIN).row(0, named=True)
    assert aspirin["CYP2D6-Direct"] == 5.0
    assert aspirin["CYP2D6-BindingDB"] == 7.0
    assert aspirin["CYP3A4-TDI"] == 4.0


def test_isoform_filter_restricts_heads(data_dir):
    _, heads, _, _ = _merge(data_dir, ["bindingdb:CYP2D6"])
    assert "CYP2D6-BindingDB" in heads and "CYP1A2-BindingDB" not in heads


def test_external_only_molecules_are_train_and_keep_smiles(data_dir):
    table, _, _, _ = _merge(data_dir, ["bindingdb"])
    toluene = table.filter(pl.col("SMILES") == TOLUENE).row(0, named=True)
    assert toluene["split"] == "train"
    assert toluene["CYP2D6-Direct"] is None


def test_source_without_labels_adds_no_rows(data_dir):
    table, _, _, _ = _merge(data_dir, ["chembl"])
    assert "CCO" in table["SMILES"].to_list()  # the only chembl-labelled molecule
    assert TOLUENE not in table["SMILES"].to_list()  # bindingdb-only row must not leak in


def test_inhouse_only_molecule_is_train(data_dir):
    table, _, _, _ = _merge(data_dir, ["tdi"])
    phenol = table.filter(pl.col("SMILES") == PHENOL).row(0, named=True)
    assert phenol["split"] == "train"


def test_split_ignores_dataset_choice(data_dir):
    held = lambda specs: sorted(  # noqa: E731
        _merge(data_dir, specs)[0].filter(pl.col("split") != "train")["SMILES"].to_list()
    )
    assert held([]) == held(["tdi", "bindingdb", "chembl"])


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        parse_dataset_specs(["nope"])


def _stability_files(data_dir):
    (data_dir / "metabolism").mkdir()
    pl.DataFrame({"SMILES": [ASPIRIN, ASPIRIN, TOLUENE, "CCO"], "Y": [1, 0, 0, 1]}).write_csv(
        data_dir / "metabolism" / "tdc_hlm.csv"
    )
    pl.DataFrame(
        {"SMILES": [ASPIRIN, IBUPROFEN], "HLM CLint": [9.0, 0.0], "MLM CLint": [99.0, None]}
    ).write_csv(data_dir / "metabolism" / "expansionrx.csv")


def test_binary_source_is_classification_and_settles_ties(data_dir):
    _stability_files(data_dir)
    table, _, _, tasks = _merge(data_dir, ["tdc_hlm"])
    assert tasks["HLM-TDC"] == "classification" and tasks["CYP2D6-Direct"] == "regression"
    aspirin = table.filter(pl.col("SMILES") == ASPIRIN).row(0, named=True)
    assert aspirin["HLM-TDC"] == 1.0  # duplicate labels 1 and 0 average to 0.5, ties go positive


def test_clint_is_log_scaled(data_dir):
    _stability_files(data_dir)
    table, heads, _, tasks = _merge(data_dir, ["expansionrx"])
    assert {"HLM-ExpansionRx", "MLM-ExpansionRx"} <= set(heads)
    assert tasks["HLM-ExpansionRx"] == "regression"
    ibuprofen = table.filter(pl.col("SMILES") == IBUPROFEN).row(0, named=True)
    assert ibuprofen["HLM-ExpansionRx"] == 0.0  # log10(0 + 1)
    aspirin = table.filter(pl.col("SMILES") == ASPIRIN).row(0, named=True)
    assert abs(aspirin["HLM-ExpansionRx"] - 1.0) < 1e-9  # log10(9 + 1)


def test_specs_can_omit_challenge():
    assert parse_dataset_specs(["tdi"], with_challenge=False) == {"tdi": None}
    assert list(parse_dataset_specs(["tdi"])) == ["challenge", "tdi"]
