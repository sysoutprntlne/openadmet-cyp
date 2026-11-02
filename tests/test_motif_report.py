"""The citation gate and the generated motif report."""

import polars as pl
import pytest

from data_tools.motif_report import build_report, uncited_entries


def test_no_entry_is_uncited_in_the_shipped_tables():
    assert uncited_entries() == []


def test_uncited_entry_is_detected(monkeypatch):
    """The gate must catch an uncited motif, not paper over it."""
    import representations.cyp_pharmacophore as cp

    blank = cp.Motif("heme_test_uncited", "n1ccccc1", "heme", "", "pyridine")
    monkeypatch.setattr(cp, "_MOTIFS", cp._MOTIFS + (blank,))
    assert "heme_test_uncited" in uncited_entries()


def test_build_report_raises_while_any_entry_is_uncited(monkeypatch):
    import representations.cyp_pharmacophore as cp

    blank = cp.Motif("heme_test_uncited", "n1ccccc1", "heme", "", "pyridine")
    monkeypatch.setattr(cp, "_MOTIFS", cp._MOTIFS + (blank,))
    train = pl.DataFrame({"SMILES": ["c1ccncc1"]})
    test = pl.DataFrame({"SMILES": ["c1ccccc1"]})
    with pytest.raises(ValueError, match="uncited"):
        build_report(train, test)


def test_report_lists_every_motif_with_smarts_source_and_prevalence():
    train = pl.DataFrame({"SMILES": ["c1ccncc1", "c1ccccc1", "c1c[nH]cn1"]})
    test = pl.DataFrame({"SMILES": ["c1ccncc1", "c1ccccc1"]})
    report = build_report(train, test)

    from representations.cyp_pharmacophore import _ISOFORM_FEATURES, _MOTIFS

    for m in _MOTIFS:
        assert m.name in report
        assert m.smarts in report
    for d in _ISOFORM_FEATURES:
        assert d.name in report

    # pyridine: 1 of 3 train molecules, 1 of 2 test molecules
    assert "33.3" in report
    assert "50.0" in report


def test_report_prevalence_ignores_invalid_rows_rather_than_counting_them_absent():
    """An unparseable SMILES is an all-NaN row; it must not dilute prevalence."""
    train = pl.DataFrame({"SMILES": ["c1ccncc1", "not_a_smiles"]})
    test = pl.DataFrame({"SMILES": ["c1ccncc1"]})
    report = build_report(train, test)
    # 1 valid of 1 parseable train molecule -> 100.0, not 50.0
    assert "100.0" in report
