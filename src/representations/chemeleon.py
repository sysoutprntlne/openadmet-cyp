"""CheMeleon pretrained foundation-model representation.

Fixed encoder only (no fine-tuning): downloads the CheMeleon message-passing
checkpoint and uses it to fingerprint SMILES into 2048-dim embeddings.
Reference: https://github.com/JacksonBurns/chemeleon (chemeleon_fingerprint.py)
"""

import os
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

import numpy as np
import polars as pl
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

from representations.base import Representation

_CHECKPOINT_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
# Relative to the working directory, like data/features, so it lands on project
# storage rather than a quota-limited home. Override with CHEMPROP_HOME.
_CHECKPOINT_DIR = Path(os.environ.get("CHEMPROP_HOME") or ".chemeleon")
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "chemeleon_mp.pt"
_OUTPUT_DIM = 2048
_BATCH_SIZE = 256


def _load_model():
    """Download (if needed) and load the CheMeleon MPNN as a fingerprint-only model."""
    import torch
    from chemprop import nn
    from chemprop.models import MPNN
    from chemprop.nn import RegressionFFN

    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not _CHECKPOINT_PATH.exists():
        urlretrieve(_CHECKPOINT_URL, _CHECKPOINT_PATH)

    checkpoint = torch.load(_CHECKPOINT_PATH, weights_only=True)
    mp = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
    mp.load_state_dict(checkpoint["state_dict"])
    model = MPNN(
        message_passing=mp,
        agg=nn.MeanAggregation(),
        predictor=RegressionFFN(input_dim=mp.output_dim),  # unused, MPNN requires one
    )
    model.eval()
    return model


class CheMeleon(Representation):
    """2048-dim molecule embeddings from the pretrained CheMeleon message-passing net.

    Invalid SMILES produce an all-NaN row.
    """

    name: ClassVar[str] = "chemeleon"

    def __init__(self) -> None:
        self._model = None
        self._featurizer = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from chemprop import featurizers

            self._model = _load_model()
            self._featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 2048) float64 array of CheMeleon embeddings."""
        import torch
        from chemprop.data import BatchMolGraph

        self._ensure_loaded()
        smiles_list = smiles.to_list()
        out = np.full((len(smiles_list), _OUTPUT_DIM), np.nan, dtype=np.float64)

        for start in tqdm(range(0, len(smiles_list), _BATCH_SIZE), desc="CheMeleon embeddings", unit="batch"):
            chunk = smiles_list[start : start + _BATCH_SIZE]
            mols = [MolFromSmiles(smi) for smi in chunk]
            valid_idx = [i for i, mol in enumerate(mols) if mol is not None]
            if not valid_idx:
                continue
            bmg = BatchMolGraph([self._featurizer(mols[i]) for i in valid_idx])
            with torch.no_grad():
                embeddings = self._model.fingerprint(bmg).numpy(force=True)
            for row, i in enumerate(valid_idx):
                out[start + i] = embeddings[row]

        return out
