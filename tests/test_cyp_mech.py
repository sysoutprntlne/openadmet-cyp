# tests/test_cyp_mech.py
"""Contract checks for the ported mechanistic block, not accuracy tests."""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("dimorphite_dl")

from representations.cyp_mech import FEATURES, CYPMechanisticBlock, most_basic_pka  # noqa: E402
from rdkit import Chem  # noqa: E402


def _row(smiles):
    block = CYPMechanisticBlock()
    return dict(zip(block.feature_names, block.transform(pl.Series([smiles]))[0]))


def test_thirty_unique_columns():
    assert len(FEATURES) == 30 and len(set(FEATURES)) == 30


def test_invalid_smiles_gives_nan_row():
    out = CYPMechanisticBlock().transform(pl.Series(["not a smiles", "", "CCN"]))
    assert out.shape == (3, 30)
    assert np.isnan(out[:2]).all()
    assert np.isfinite(out[2]).all()


@pytest.mark.parametrize("smiles,lit,tol", [
    ("CCN", 10.7, 1.0),
    ("c1ccncc1", 5.2, 0.5),
    ("C1CCNCC1", 11.1, 1.0),
])
def test_pka_near_literature(smiles, lit, tol):
    pka, _, _ = most_basic_pka(Chem.MolFromSmiles(smiles))
    assert abs(pka - lit) <= tol


def test_diphenhydramine_matches_cyp2d6_pharmacophore():
    r = _row("CN(C)CCOC(c1ccccc1)c1ccccc1")
    assert r["is_base_74"] == 1.0
    assert r["pharm_2d6"] == 1.0
    assert r["n_amine_tert"] == 1.0
    assert r["ph74_n_cation"] == 1.0


def test_acid_has_no_basic_centre():
    r = _row("CC(=O)O")
    assert r["n_acid"] == 1.0
    assert r["is_base_74"] == 0.0 and r["pharm_2d6"] == 0.0
    assert r["ph74_n_anion"] == 1.0
