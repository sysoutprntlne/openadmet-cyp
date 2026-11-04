
"""RDKit 3D molecular shape descriptor representation."""

from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from tqdm import tqdm

from representations.base import Representation


# Descriptors calculated from a molecule's 3D conformer.
DESCRIPTOR_NAMES = [
    "PMI1",
    "PMI2",
    "PMI3",
    "NPR1",
    "NPR2",
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "RadiusOfGyration",
    "SpherocityIndex",
    "PBF",
]


class ShapeDescriptors(Representation):
    """Generate deterministic RDKit 3D shape features."""

    name: ClassVar[str] = "shape"

    @property
    def feature_names(self) -> list[str]:
        return DESCRIPTOR_NAMES.copy()

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 11) float64 feature matrix."""

        out = np.full(
            (len(smiles), len(DESCRIPTOR_NAMES)),
            np.nan,
            dtype=np.float64,
        )

        for i, smi in enumerate(
            tqdm(smiles.to_list(), desc="3D shape descriptors", unit="mol")
        ):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                mol = Chem.AddHs(mol)

                # Fixed seed ensures reproducible conformer generation.
                params = AllChem.ETKDGv3()
                params.randomSeed = 42

                if AllChem.EmbedMolecule(mol, params) != 0:
                    continue

                # Optimize the generated 3D geometry.
                if not AllChem.MMFFHasAllMoleculeParams(mol):
                    continue

                if AllChem.MMFFOptimizeMolecule(mol, maxIters=1000) != 0:
                    continue

                # Calculate 3D descriptors.
                values = [
                    getattr(Descriptors3D, name)(mol)
                    for name in DESCRIPTOR_NAMES
                ]

                if np.all(np.isfinite(values)):
                    out[i] = values

            except (ValueError, RuntimeError, TypeError):
                continue

        return out
