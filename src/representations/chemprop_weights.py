"""Embeddings from a trained chemprop model's message-passing encoder.

Selected as `chemprop[/path/to/weights]` (or `chemprop[alias]`, see foundation_alias) wherever an input name is accepted.
The weights can be a chemprop model file (`best*.pt`, as written by
`chemprop train` / `train-chemprop`), a Lightning `.ckpt`, or a message-passing-only
checkpoint like CheMeleon's. The fingerprint is the aggregated (and batch-normed)
graph embedding; the prediction head is not used.
"""

import re
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

from representations.base import Representation
from representations.foundation_alias import resolve_alias

_BATCH_SIZE = 256
_SPEC = re.compile(r"^chemprop\[(.+)\]$")


def parse_spec(name: str) -> Path | None:
    """Return the weights path of a `chemprop[/path]` or `chemprop[alias]` input name, or None if it isn't one."""
    match = _SPEC.match(name)
    return Path(resolve_alias(match.group(1))).expanduser() if match else None


def cache_name(name: str) -> str:
    """Filesystem-safe cache stem for a spec, so the cache is keyed by the weights, not just 'chemprop'.

    Includes size and mtime: retraining to the same path must not reuse stale embeddings.
    """
    path = parse_spec(name)
    if path is None:
        raise ValueError(f"not a chemprop[...] input: {name!r}")
    stat = path.stat()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(path.resolve())).strip("_")[-80:]
    return f"chemprop_{slug}_{stat.st_size}_{int(stat.st_mtime)}"


def _load_model(path: Path):
    """Load a full chemprop model, or wrap a message-passing-only checkpoint as a fingerprint model."""
    import torch
    from chemprop import nn
    from chemprop.models import MPNN
    from chemprop.nn import RegressionFFN

    if not path.exists():
        raise FileNotFoundError(f"chemprop weights not found: {path}")
    if path.suffix == ".ckpt":
        model = MPNN.load_from_checkpoint(path, map_location="cpu")
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "message_passing" in checkpoint["hyper_parameters"]:
            model = MPNN.load_from_file(path, map_location="cpu")
        else:
            mp = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
            mp.load_state_dict(checkpoint["state_dict"])
            model = MPNN(
                message_passing=mp,
                agg=nn.MeanAggregation(),
                predictor=RegressionFFN(input_dim=mp.output_dim),  # unused, MPNN requires one
            )
    model.eval()
    return model


class ChempropWeights(Representation):
    """Graph embeddings from the encoder of the chemprop model at `path`. Invalid SMILES give an all-NaN row."""

    name: ClassVar[str] = "chemprop"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._model = None
        self._featurizer = None

    def transform(self, smiles: pl.Series) -> np.ndarray:
        import torch
        from chemprop import featurizers
        from chemprop.data import BatchMolGraph

        if self._model is None:
            self._model = _load_model(self.path)
            self._featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        smiles_list = smiles.to_list()
        out: np.ndarray | None = None
        for start in tqdm(range(0, len(smiles_list), _BATCH_SIZE), desc="chemprop embeddings", unit="batch"):
            chunk = smiles_list[start : start + _BATCH_SIZE]
            mols = [MolFromSmiles(smi) for smi in chunk]
            valid_idx = [i for i, mol in enumerate(mols) if mol is not None]
            if not valid_idx:
                continue
            bmg = BatchMolGraph([self._featurizer(mols[i]) for i in valid_idx])
            with torch.no_grad():
                embeddings = self._model.fingerprint(bmg).numpy(force=True)
            if out is None:
                out = np.full((len(smiles_list), embeddings.shape[1]), np.nan, dtype=np.float64)
            for row, i in enumerate(valid_idx):
                out[start + i] = embeddings[row]
        if out is None:
            raise ValueError("no valid SMILES to embed with the chemprop model")
        return out
