"""RDKit MACCS structural-key representation."""

from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from tqdm import tqdm

from representations.base import Representation

_N_FEATURES = 167


class MACCS(Representation):
    """167-position MACCS fingerprint, including RDKit's unused position zero.

    Invalid, null, and blank SMILES produce an all-NaN row.
    """

    name: ClassVar[str] = "maccs"

    @property
    def feature_names(self) -> list[str]:
        """Return names in RDKit bit order, including unused MACCS_0."""
        return [f"MACCS_{i}" for i in range(_N_FEATURES)]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 167) float64 matrix in input row order."""
        out = np.full((len(smiles), _N_FEATURES), np.nan, dtype=np.float64)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc="MACCS keys", unit="mol", leave=True)):
            if smi is None or not smi.strip():
                continue
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                out[i] = np.asarray(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.float64)
        return out
