# tests/test_cyp_xtb.py
"""Contract checks for the GFN2-xTB blocks, not accuracy tests."""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("tblite")

from representations import cyp_xtb  # noqa: E402
from representations.cyp_xtb import (  # noqa: E402
    ALL_FEATURES,
    ELEC_FEATURES,
    SOLV_FEATURES,
    XTBElectronic,
    XTBSolvation,
)


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("XTB_NJOBS", "1")
    monkeypatch.setattr(cyp_xtb, "_MEMO", {})


def _row(smiles):
    return dict(zip(ALL_FEATURES, cyp_xtb._compute_one(smiles, n_confs=3)))


def test_shapes_and_names():
    smi = pl.Series(["Cc1ccncc1", "CC(=O)O"])
    solv = XTBSolvation().transform(smi)
    elec = XTBElectronic().transform(smi)
    assert solv.shape == (2, len(SOLV_FEATURES))
    assert elec.shape == (2, len(ELEC_FEATURES))
    assert len(XTBSolvation().feature_names) == solv.shape[1]
    assert np.isfinite(solv).all() and np.isfinite(elec).all()


def test_invalid_smiles_gives_nan_row():
    out = XTBElectronic().transform(pl.Series(["not_a_smiles", "", "CCO"]))
    assert np.isnan(out[0]).all() and np.isnan(out[1]).all()
    assert np.isfinite(out[2]).all()


def test_ionization_direction():
    acid, neutral, amine = _row("CC(=O)O"), _row("CCCC"), _row("CCN(CC)CC")
    assert acid["log_fneutral_ph74"] < -2 and acid["n_acid_groups"] == 1
    assert amine["log_fneutral_ph74"] < -1 and amine["n_base_groups"] == 1
    assert neutral["log_fneutral_ph74"] == 0.0
    assert acid["logd_xtb"] == pytest.approx(acid["logp_xtb"] + acid["log_fneutral_ph74"])


def test_lipophilicity_ordering():
    assert _row("c1ccccc1CCCCCC")["logp_xtb"] > _row("OCC(O)CO")["logp_xtb"]


def test_deterministic_and_salt_stripped():
    a, b = _row("CCN"), _row("CCN.Cl")
    for k in ALL_FEATURES:
        assert a[k] == pytest.approx(b[k], rel=1e-6, abs=1e-8)
    assert _row("CCN") == a


def test_registered():
    from data_tools.inputs import INPUT_REGISTRY

    assert {"xtb_solv_v1", "xtb_elec_v1"} <= set(INPUT_REGISTRY)
