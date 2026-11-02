"""Uni-Mol molecular embeddings.

Uses a Uni-Mol v1 encoder to generate 512-dimensional molecular representations from SMILES. By default this is
the pretrained encoder, selected as `unimol`. A fine-tuned model is selected as `unimol[/path/to/model_0.pth]`
(or `unimol[alias]`, see foundation_alias) wherever an input name is accepted: the weights are the `model_0.pth`
written by `train-unimol`, and the fingerprint is the encoder's CLS embedding; the regression head is not used.
"""

import os
import re
from pathlib import Path
from typing import ClassVar
from rdkit.Chem import AllChem

import numpy as np
import polars as pl
from rdkit import Chem
from tqdm import tqdm

from representations.base import Representation
from representations.foundation_alias import resolve_alias


_OUTPUT_DIM = 512
_BATCH_SIZE = 32
_SPEC = re.compile(r"^unimol\[(.+)\]$")


def parse_spec(name: str) -> Path | None:
    """Return the weights path of a `unimol[/path]` or `unimol[alias]` input name, or None if it isn't one."""
    match = _SPEC.match(name)
    return Path(resolve_alias(match.group(1))).expanduser() if match else None


def cache_name(name: str) -> str:
    """Filesystem-safe cache stem for a spec, so the cache is keyed by the weights, not just 'unimol'.

    Includes size and mtime: retraining to the same path must not reuse stale embeddings.
    """
    path = parse_spec(name)
    if path is None:
        raise ValueError(f"not a unimol[...] input: {name!r}")
    stat = path.stat()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(path.resolve())).strip("_")[-80:]
    return f"unimol_{slug}_{stat.st_size}_{int(stat.st_mtime)}"


def _dict_path(weights: Path) -> str:
    """The token dictionary for fine-tuned weights: dict.txt beside them if present, else the pretrained one."""
    beside = weights.parent / "dict.txt"
    if beside.is_file():
        return str(beside)
    from unimol_tools.config import MODEL_CONFIG
    from unimol_tools.weights import get_weight_dir, weight_download

    name = MODEL_CONFIG["dict"]["molecule_all_h"]
    if not (Path(get_weight_dir()) / name).exists():
        weight_download(name, get_weight_dir())
    return os.path.join(get_weight_dir(), name)


def _encoder_only(path: Path) -> Path:
    """Copy of a fine-tuned checkpoint without its output layer, which UniMolRepr's 1-output head can't load.

    Fine-tuning on several targets leaves a (n_targets, 512) out_proj; the embeddings come from the encoder alone.
    """
    import tempfile

    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state = {k: v for k, v in state.items() if not k.startswith("classification_head.out_proj.")}
    stripped = Path(tempfile.mkdtemp()) / path.name
    torch.save({"model_state_dict": state} if "model_state_dict" in checkpoint else state, stripped)
    return stripped


class UniMol(Representation):
    """Generate 512-dimensional embeddings using Uni-Mol: pretrained, or the fine-tuned model at `path`."""

    name: ClassVar[str] = "unimol"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._model = None 

    def _ensure_loaded(self) -> None:
        """Load the pretrained Uni-Mol model only when needed."""
        if self._model is None:
            import torch
            from unimol_tools import UniMolRepr

            weights = {}
            if self.path is not None:
                if not self.path.exists():
                    raise FileNotFoundError(f"unimol weights not found: {self.path}")
                weights = {
                    "pretrained_model_path": str(_encoder_only(self.path)),
                    "pretrained_dict_path": _dict_path(self.path),
                }
            self._model = UniMolRepr(
                data_type="molecule",
                model_name="unimolv1",
                use_cuda=torch.cuda.is_available(),
                **weights,
            )

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return deterministic 512-dimensional Uni-Mol embeddings."""
        self._ensure_loaded()

        smiles_list = smiles.to_list()
        out = np.full(
            (len(smiles_list), _OUTPUT_DIM),
            np.nan,
            dtype=np.float64,
        )

        failed_conformers = 0

        for start in tqdm(
            range(0, len(smiles_list), _BATCH_SIZE),
            desc="Uni-Mol embeddings",
            unit="batch",
        ):
            chunk = smiles_list[start:start + _BATCH_SIZE]

            valid_idx = []
            atoms_list = []
            coordinates_list = []

            for i, smi in enumerate(chunk):
                if not isinstance(smi, str):
                    continue

                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                try:
                    mol = Chem.AddHs(mol)

                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42

                    if AllChem.EmbedMolecule(mol, params) != 0:
                        failed_conformers += 1
                        continue

                    if not AllChem.MMFFHasAllMoleculeParams(mol):
                        failed_conformers += 1
                        continue

                    if AllChem.MMFFOptimizeMolecule(
                        mol, maxIters=1000
                    ) != 0:
                        failed_conformers += 1
                        continue

                    atoms = [
                        atom.GetSymbol()
                        for atom in mol.GetAtoms()
                    ]

                    coordinates = (
                        mol.GetConformer()
                        .GetPositions()
                        .astype(np.float32)
                    )

                    if not np.isfinite(coordinates).all():
                        failed_conformers += 1
                        continue

                    valid_idx.append(i)
                    atoms_list.append(atoms)
                    coordinates_list.append(coordinates)

                except (ValueError, RuntimeError, TypeError):
                    failed_conformers += 1
                    continue

            if not valid_idx:
                continue

            embeddings = np.asarray(
                self._model.get_repr({
                    "atoms": atoms_list,
                    "coordinates": coordinates_list,
                }),
                dtype=np.float64,
            )

            if embeddings.shape != (len(valid_idx), _OUTPUT_DIM):
                raise ValueError(
                    f"Unexpected Uni-Mol embedding shape: {embeddings.shape}"
                )

            for row, i in enumerate(valid_idx):
                if np.isfinite(embeddings[row]).all():
                    out[start + i] = embeddings[row]

        print(f"Failed 3D conformers: {failed_conformers}/{len(smiles_list)}")

        return out