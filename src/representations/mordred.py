"""2D Mordred descriptors using the optional mordredcommunity package."""

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from tqdm import tqdm

from representations.base import Representation

if TYPE_CHECKING:
    from mordred import Calculator


class MordredDescriptors(Representation):
    """All configured 2D descriptors in a fixed calculator-defined order.

    Invalid, null, and blank SMILES produce an all-NaN row. For valid molecules,
    missing/error and string descriptor values are encoded as zero, following
    DeepChem's Mordred featurizer convention; non-finite numbers also become
    zero. This encoding does not distinguish missing values from true zeros.
    Descriptor columns are never dropped or reordered.
    The calculator is loaded on first use and reused for later calls.

    Existing NaN-valued feature caches must be rebuilt or a fresh cache used.
    Reference: https://github.com/deepchem/deepchem/blob/master/deepchem/feat/
    molecule_featurizers/mordred_descriptors.py
    """

    name: ClassVar[str] = "mordred"

    def __init__(self) -> None:
        self._calculator: "Calculator | None" = None

    def _get_calculator(self) -> "Calculator":
        """Load the optional dependency and initialize the 2D calculator once."""
        if self._calculator is None:
            try:
                from mordred import Calculator, descriptors
            except ModuleNotFoundError as exc:
                if exc.name != "mordred":
                    raise
                raise ImportError(
                    "Mordred descriptors require mordredcommunity. "
                    "Install it with: python -m pip install mordredcommunity"
                ) from exc
            self._calculator = Calculator(descriptors, ignore_3D=True)
        return self._calculator

    @property
    def feature_names(self) -> list[str]:
        """Return descriptor names in the same order as the feature columns."""
        return [str(descriptor) for descriptor in self._get_calculator().descriptors]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, n_descriptors) float64 matrix in input order."""
        calculator = self._get_calculator()
        out = np.full((len(smiles), len(calculator.descriptors)), np.nan, dtype=np.float64)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc="Mordred descriptors", unit="mol", leave=True)):
            if smi is None or not smi.strip():
                continue
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                result = calculator(mol).fill_missing(0.0)
                row = np.asarray([0.0 if isinstance(value, str) else value for value in result], dtype=np.float64)
                row[~np.isfinite(row)] = 0.0
                out[i] = row
        return out
