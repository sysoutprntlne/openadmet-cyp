# tests/test_cyp_pharmacophore.py
"""Synthetic contract checks for the CYP pharmacophore blocks, not accuracy tests."""

import numpy as np
import polars as pl
import pytest

from representations.cyp_pharmacophore import (
    _MOTIFS,
    HemeCoordinationMotifs,
)

HEME = HemeCoordinationMotifs()


def _row(block, smiles):
    """Return the single feature row for one SMILES as a dict keyed by column name."""
    arr = block.transform(pl.Series([smiles]))
    return dict(zip(block.feature_names, arr[0]))


def test_every_motif_smarts_compiles():
    from rdkit import Chem

    bad = [m.name for m in _MOTIFS if Chem.MolFromSmarts(m.smarts) is None]
    assert bad == []


def test_every_motif_rollup_is_a_t6_donor_label():
    allowed = {"imidazole", "triazole", "pyridine", "aliphatic amine"}
    for m in _MOTIFS:
        if m.block == "heme" and m.rollup:
            assert m.rollup in allowed, f"{m.name} has rollup {m.rollup!r}"


def test_feature_names_match_matrix_width():
    arr = HEME.transform(pl.Series(["c1ccccc1"]))
    assert arr.shape == (1, len(HEME.feature_names))


# --- the Podgorski pair: the sharpest available test of the hindrance term ---
# 4-(pyridin-3-yl)benzoic acid coordinates heme iron; 4-(pyridin-2-yl)benzoic
# acid does not (an aqua ligand remains). Crystallographically resolved.
# Podgorski et al. 2020, Biochemistry — spec section 12.1.
def test_podgorski_pair_separates_on_hindrance():
    coordinates = _row(HEME, "OC(=O)c1ccc(cc1)-c1cccnc1")   # pyridin-3-yl
    does_not = _row(HEME, "OC(=O)c1ccc(cc1)-c1ccccn1")      # pyridin-2-yl

    assert coordinates["heme_donor_unhindered"] == 1.0
    assert coordinates["heme_donor_hindered"] == 0.0

    assert does_not["heme_donor_unhindered"] == 0.0
    assert does_not["heme_donor_hindered"] == 1.0

    # both are pyridines; only accessibility differs
    assert coordinates["heme_pyridine"] == does_not["heme_pyridine"] == 1.0


def test_ortho_disubstituted_pyridine_is_hindered():
    r = _row(HEME, "Cc1cccc(C)n1")  # 2,6-dimethylpyridine
    assert r["heme_donor_hindered"] == 1.0
    assert r["heme_donor_unhindered"] == 0.0


def test_known_type_ii_inhibitors_have_an_unhindered_donor():
    keto = _row(HEME, "CC(=O)N1CCN(CC1)c1ccc(OC[C@H]2CO[C@@](Cn3ccnc3)(O2)c2ccc(Cl)cc2Cl)cc1")
    assert keto["heme_imidazole"] == 1.0
    assert keto["heme_donor_unhindered"] >= 1.0

    flu = _row(HEME, "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F")
    assert flu["heme_triazole_124"] == 2.0
    assert flu["heme_donor_unhindered"] == 4.0


def test_tetrazole_is_counted():
    losartan = _row(HEME, "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1")
    assert losartan["heme_tetrazole"] == 1.0


def test_pyrazole_is_not_counted_as_imidazole():
    assert _row(HEME, "c1cc[nH]n1")["heme_imidazole"] == 0.0
    assert _row(HEME, "c1c[nH]cn1")["heme_imidazole"] == 1.0


def test_primary_aliphatic_amine_counted_but_not_amide_or_aniline():
    assert _row(HEME, "CC(N)Cc1ccccc1")["heme_amine_primary"] == 1.0   # amphetamine
    assert _row(HEME, "CC(N)=O")["heme_amine_primary"] == 0.0          # acetamide
    assert _row(HEME, "Nc1ccccc1")["heme_amine_primary"] == 0.0        # aniline


# --- Review Focus items ---
def test_no_match_molecule_gives_zeros_not_nan():
    """Critical: NaN here would make drop_nan_rows delete every non-azole molecule."""
    arr = HEME.transform(pl.Series(["c1ccccc1"]))
    assert not np.isnan(arr).any()
    assert (arr == 0).all()


@pytest.mark.parametrize("bad", [None, "", "   ", "not_a_smiles"])
def test_invalid_null_and_blank_smiles_give_all_nan(bad):
    """MolFromSmiles('') returns a VALID zero-atom mol, so 'mol is not None' is not enough."""
    arr = HEME.transform(pl.Series([bad], dtype=pl.Utf8))
    assert np.isnan(arr).all()


def test_empty_input_returns_zero_rows_with_full_width():
    arr = HEME.transform(pl.Series([], dtype=pl.Utf8))
    assert arr.shape == (0, len(HEME.feature_names))


def test_duplicate_smiles_return_one_row_each_in_input_order():
    arr = HEME.transform(pl.Series(["c1ccccc1", "c1ccncc1", "c1ccccc1"]))
    assert arr.shape[0] == 3
    np.testing.assert_array_equal(arr[0], arr[2])
    assert arr[1][HEME.feature_names.index("heme_pyridine")] == 1.0


def test_transform_is_deterministic():
    s = pl.Series(["c1ccncc1", "CC(N)Cc1ccccc1"])
    np.testing.assert_array_equal(HEME.transform(s), HEME.transform(s))


def test_dtype_is_float64():
    assert HEME.transform(pl.Series(["c1ccncc1"])).dtype == np.float64


# --- mechanism-based inactivation (cyp_tdi_v2) ---
from representations.cyp_pharmacophore import MechanismBasedAlerts

TDI = MechanismBasedAlerts()


def test_tdi_block_is_separate_from_heme_block():
    assert set(TDI.feature_names).isdisjoint(HEME.feature_names)
    assert TDI.name != HEME.name


def test_methylenedioxyphenyl_alert():
    assert _row(TDI, "C=CCc1ccc2OCOc2c1")["tdi_mdp"] == 1.0  # safrole
    assert _row(TDI, "Fc1ccc(cc1)[C@@H]1CCNC[C@H]1COc1ccc2OCOc2c1")["tdi_mdp"] == 1.0  # paroxetine
    assert _row(TDI, "c1ccccc1")["tdi_mdp"] == 0.0


def test_terminal_alkyne_alert_excludes_internal_alkynes():
    ethinyl = "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@]2(O)C#C"
    assert _row(TDI, ethinyl)["tdi_terminal_alkyne"] == 1.0  # ethinylestradiol
    assert _row(TDI, "CC#C")["tdi_terminal_alkyne"] == 1.0   # propyne
    assert _row(TDI, "CC#CC")["tdi_terminal_alkyne"] == 0.0  # 2-butyne, internal


def test_furan_and_thiophene_alerts():
    assert _row(TDI, "c1ccoc1")["tdi_furan"] == 1.0
    assert _row(TDI, "Clc1ccccc1CN1CCc2sccc2C1")["tdi_thiophene"] == 1.0  # ticlopidine
    assert _row(TDI, "c1ccccc1")["tdi_furan"] == 0.0


def test_hydrazine_alert():
    assert _row(TDI, "NNC(=O)c1ccncc1")["tdi_hydrazine"] == 1.0  # isoniazid


def test_tdi_no_match_gives_zeros_and_invalid_gives_nan():
    assert not np.isnan(TDI.transform(pl.Series(["c1ccccc1"]))).any()
    assert np.isnan(TDI.transform(pl.Series([""], dtype=pl.Utf8))).all()


# --- isoform-preference descriptors (cyp_isoform_v2) ---
from representations.cyp_pharmacophore import (
    _ISOFORM_FEATURES,
    IsoformPreferenceDescriptors,
)

ISO = IsoformPreferenceDescriptors()


def test_every_isoform_descriptor_is_cited():
    uncited = [d.name for d in _ISOFORM_FEATURES if not d.source.strip()]
    assert uncited == []


def test_isoform_width_matches_names():
    arr = ISO.transform(pl.Series(["c1ccncc1"]))
    assert arr.shape == (1, len(ISO.feature_names))


def test_basic_amine_count_tracks_basicity():
    """CYP2D6 ligands characteristically carry a basic nitrogen (Paine 2003)."""
    assert _row(ISO, "CC(C)NCC(O)COc1cccc2ccccc12")["iso_basic_amine_count"] == 1.0  # propranolol
    assert _row(ISO, "c1ccccc1")["iso_basic_amine_count"] == 0.0


def test_acidic_group_count_tracks_anionic_ligands():
    """CYP2C9 prefers anionic ligands via Arg108 (Wester 2004, Dickmann 2004)."""
    assert _row(ISO, "CC(C)Cc1ccc(cc1)C(C)C(=O)O")["iso_acidic_group_count"] == 1.0  # ibuprofen
    assert _row(ISO, "c1ccccc1")["iso_acidic_group_count"] == 0.0


def test_aromatic_fraction_is_a_planarity_proxy():
    """CYP1A2's cavity is adapted to large planar substrates (Sansen 2007)."""
    flat = _row(ISO, "c1ccc2ccccc2c1")["iso_aromatic_frac"]      # naphthalene
    saturated = _row(ISO, "CC1CCCCCCC1")["iso_aromatic_frac"]    # cyclooctane
    assert flat == 1.0
    assert saturated == 0.0


def test_size_and_lipophilicity_present_and_finite():
    r = _row(ISO, "CC(C)NCC(O)COc1cccc2ccccc12")
    assert 250.0 < r["iso_mw"] < 270.0
    assert np.isfinite(r["iso_clogp"])
    assert r["iso_aromatic_rings"] == 2.0


def test_isoform_invalid_smiles_gives_all_nan():
    for bad in [None, "", "   ", "not_a_smiles"]:
        assert np.isnan(ISO.transform(pl.Series([bad], dtype=pl.Utf8))).all()


def test_isoform_hbond_and_polarity_columns_are_not_transposed():
    """iso_hba and iso_hbd are asymmetric for propranolol, so a swap is caught."""
    prop = _row(ISO, "CC(C)NCC(O)COc1cccc2ccccc12")
    assert prop["iso_hba"] == 3.0
    assert prop["iso_hbd"] == 2.0
    assert prop["iso_tpsa"] == pytest.approx(41.49, abs=0.01)
    assert prop["iso_fraction_csp3"] == pytest.approx(0.375, abs=0.001)

    flat = _row(ISO, "c1ccccc1")
    assert flat["iso_hba"] == 0.0
    assert flat["iso_hbd"] == 0.0
    assert flat["iso_tpsa"] == 0.0
    assert flat["iso_fraction_csp3"] == 0.0


from representations.cyp_pharmacophore import ERG_N_FEATURES, ErGFingerprint

ERG = ErGFingerprint()


def test_erg_width_is_315():
    assert ERG_N_FEATURES == 315
    arr = ERG.transform(pl.Series(["CC(C)NCC(O)COc1cccc2ccccc12"]))
    assert arr.shape == (1, 315)
    assert len(ERG.feature_names) == 315


def test_erg_is_not_all_zero_for_a_real_molecule():
    arr = ERG.transform(pl.Series(["CC(C)NCC(O)COc1cccc2ccccc12"]))
    assert (arr != 0).sum() > 0


def test_erg_invalid_smiles_gives_all_nan():
    for bad in [None, "", "   ", "not_a_smiles"]:
        assert np.isnan(ERG.transform(pl.Series([bad], dtype=pl.Utf8))).all()


def test_erg_empty_input_keeps_full_width():
    assert ERG.transform(pl.Series([], dtype=pl.Utf8)).shape == (0, 315)


def test_erg_is_deterministic():
    s = pl.Series(["CC(C)NCC(O)COc1cccc2ccccc12"])
    np.testing.assert_array_equal(ERG.transform(s), ERG.transform(s))


def test_erg_row_matches_a_direct_rdkit_call():
    """Shape tests alone would pass a wrong-but-same-shape fingerprint function."""
    from rdkit import Chem
    from rdkit.Chem import rdReducedGraphs

    smi = "c1ccccc1"
    direct = np.asarray(
        rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(smi)), dtype=np.float64
    )
    np.testing.assert_array_equal(ERG.transform(pl.Series([smi]))[0], direct)

    # structurally different molecules must not produce identical vectors
    prop = ERG.transform(pl.Series(["CC(C)NCC(O)COc1cccc2ccccc12"]))[0]
    assert not np.array_equal(prop, direct)


from data_tools.inputs import INPUT_REGISTRY


def test_all_four_blocks_are_registered():
    for name in ["cyp_heme_v2", "cyp_isoform_v2", "cyp_tdi_v2", "erg"]:
        assert name in INPUT_REGISTRY, f"{name} missing from INPUT_REGISTRY"


def test_registry_entries_are_callable_and_return_correct_width():
    df = pl.DataFrame({"SMILES": ["c1ccncc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]})
    for block in [HEME, ISO, TDI, ERG]:
        arr = INPUT_REGISTRY[block.name](df)
        assert arr.shape == (2, len(block.feature_names)), block.name


def test_preexisting_registry_order_is_unchanged():
    """Registry edits are append-only; pinned results depend on the order."""
    assert list(INPUT_REGISTRY) == [
        "rdkit", "chemeleon", "morgan", "count_morgan", "maccs", "mordred", "shape", "unimol",
        "cyp_heme_v2", "cyp_isoform_v2", "cyp_tdi_v2", "erg",
        "xtb_solv_v1", "xtb_elec_v1", "cyp_mech_v1",
    ]


def test_ring_motifs_require_a_donor_nitrogen_not_merely_the_ring():
    """A pyrimidinone or pyridone nitrogen cannot ligate iron; the ring alone is not enough."""
    assert _row(HEME, "O=c1cc[nH]c(=O)[nH]1")["heme_pyrimidine"] == 0.0   # uracil
    assert _row(HEME, "O=c1cccc[nH]1")["heme_pyridine"] == 0.0            # 2-pyridone
    assert _row(HEME, "Cn1c(=O)c2c(ncn2C)n(C)c1=O")["heme_pyrimidine"] == 0.0  # caffeine
    # true positives still fire
    assert _row(HEME, "c1ccncc1")["heme_pyridine"] == 1.0
    assert _row(HEME, "c1cncnc1")["heme_pyrimidine"] == 1.0
    assert _row(HEME, "c1ccc2scnc2c1")["heme_thiazole"] == 1.0            # benzothiazole


def test_basic_amine_patterns_exclude_amidine_guanidine_and_hydrazine():
    """These nitrogens are not basic aliphatic amines, despite matching a naive NX3 pattern."""
    for smi in ["CC(=N)N(C)C", "NC(=N)N(C)C", "CC(=S)N(C)C", "CN(C)NC", "CN(C)O"]:
        r = _row(HEME, smi)
        assert r["heme_amine_basic"] == 0.0, smi
        assert r["heme_amine_primary"] == 0.0, smi


def test_tertiary_alkylamine_alert_fires_and_excludes_non_amines():
    assert _row(TDI, "CCN(CC)CC")["tdi_alkylamine_tertiary"] == 1.0   # triethylamine
    assert _row(TDI, "CC(=O)N(C)C")["tdi_alkylamine_tertiary"] == 0.0  # amide
    assert _row(TDI, "NC(=N)N(C)C")["tdi_alkylamine_tertiary"] == 0.0  # guanidine


def test_hydrazine_alert_still_fires_after_amine_tightening():
    """Tightening the amine patterns must not disturb the separate hydrazine alert."""
    assert _row(TDI, "CN(C)NC")["tdi_hydrazine"] == 1.0


def test_table_content_hash_pins_the_version():
    """Editing a SMARTS or descriptor without bumping TABLE_VERSION must fail here.

    If this test fails because you deliberately changed a table: bump TABLE_VERSION in
    src/representations/cyp_pharmacophore.py, then update the expected hash below with
    the value this failure prints. Do not update the hash without bumping the version —
    the feature cache is keyed on the registered name, so a silent table change serves
    stale columns (see the spec, section 5.1).
    """
    import hashlib

    import representations.cyp_pharmacophore as cp

    payload = "|".join(
        [f"{m.name}={m.smarts}" for m in cp._MOTIFS]
        + [d.name for d in cp._ISOFORM_FEATURES]
        + [f"ERG={cp.ERG_N_FEATURES}", f"VERSION={cp.TABLE_VERSION}"]
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    assert digest == "701eaa7e737a5088", (
        f"table changed; digest is now {digest}. Read this test's docstring."
    )
