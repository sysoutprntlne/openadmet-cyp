"""Morgan fingerprint molecular representations."""

from typing import ClassVar

import numpy as np
import polars as pl #to hold smiles
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator #gives access to rdkit fingerprint generator functions
from tqdm import tqdm

from representations.base import Representation

class MorganFingerprint(Representation):
    """2048-bit Morgan (ECFP4) fingerprint with radius 2.

    Invalid, null, and blank SMILES produce an all-NaN row.
    """

    name: ClassVar[str] = "morgan"

    def __init__(self) -> None: ##automatically runs when a new MorganFingerprint object is created. None means returns nothing
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
        )  
    def transform(self, smiles: pl.Series) -> np.ndarray: #smiles expected to be polar series; returns the numpy array
        """Return a (n_molecules, 2048) Morgan fingerprint array."""
        out = np.full((len(smiles), 2048), np.nan, dtype=np.float64)
        # np.full --> fill the array with np.nan
        # the array's dimensions are len(smiles) which is num of molecules, and each of those will have 2048 bits
        # np.nan --> not a number (not 0 because 0 means smth in morgan)
        # dtype=np.float64 --> everything in returned matrix will use numpy 64bit float, which matches Representation

        for i, smi in enumerate( #enumerate loops through items and gives the index for each item
            tqdm(smiles.to_list(), desc="Morgan fingerprint", unit="mol", leave=True)
            # smiles to list --> takes polar smiles list to normal python list
            # tqdm --> when processing thousands of molecules, terminal will show progress bar
            # for statement --> i (row number), smi (one smiles string)
        ):
            if smi is None or not smi.strip(): # null or blank SMILES keeps its all-NaN row
                continue
            mol = Chem.MolFromSmiles(smi.strip()) # converst SMILES string to RDKit's Mol representation

            if mol is not None: #not None because if RDKit cannot parse SMILE, RDKit will say "none"
                fp = self.generator.GetFingerprint(mol) #we are now using our init and feeding it to RDKit to return the 2048bit Morgan fingerprint
                out[i] = fp # have to conform to the Numpy float 64 matrix Representation requirement and store it to row i
        return out #returns finished matrix

class CountMorganFingerprint(Representation):
    """2048-dimensional count Morgan fingerprint with radius 2.

    Invalid, null, and blank SMILES produce an all-NaN row.
    """

    name: ClassVar[str] = "count_morgan"

    def __init__(self) -> None:
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
        )

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 2048) count Morgan fingerprint array."""
        out = np.full((len(smiles), 2048), np.nan, dtype=np.float64)

        for i, smi in enumerate(
            tqdm(smiles.to_list(), desc="Count-Morgan fingerprint", unit="mol", leave=True)
        ):
            if smi is None or not smi.strip():
                continue
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                fp = self.generator.GetCountFingerprintAsNumPy(mol) # only distinction from normal Morgan. Using GetCountFingerprint instead of GetFingerprint
                out[i] = fp

        return out