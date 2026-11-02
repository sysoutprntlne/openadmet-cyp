"""Registry of molecular input featurizations for the CYP challenge."""

from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from representations.chemeleon import CheMeleon
from representations.chemprop_weights import ChempropWeights, cache_name, parse_spec
from representations.rdkit_descriptors import RDKitDescriptors
from representations.morgan import CountMorganFingerprint, MorganFingerprint
from representations.maccs import MACCS
from representations.mordred import MordredDescriptors
from representations.cyp_pharmacophore import (
    ErGFingerprint,
    HemeCoordinationMotifs,
    IsoformPreferenceDescriptors,
    MechanismBasedAlerts,
)
from representations.shape import ShapeDescriptors
from representations.unimol import UniMol
from representations.unimol import cache_name as unimol_cache_name
from representations.unimol import parse_spec as parse_unimol_spec
from representations.cyp_mech import CYPMechanisticBlock
from representations.cyp_xtb import XTBElectronic, XTBSolvation

_rdkit = RDKitDescriptors()
_chemeleon = CheMeleon()
_morgan = MorganFingerprint()
_count_morgan = CountMorganFingerprint()
_maccs = MACCS()
_mordred = MordredDescriptors()
_shape = ShapeDescriptors()
_unimol = UniMol()
_cyp_heme = HemeCoordinationMotifs()
_cyp_isoform = IsoformPreferenceDescriptors()
_cyp_tdi = MechanismBasedAlerts()
_erg = ErGFingerprint()
_xtb_solv = XTBSolvation()
_xtb_elec = XTBElectronic()
_cyp_mech = CYPMechanisticBlock()


def _shape_features(df: pl.DataFrame) -> np.ndarray:
    return _shape.transform(df["SMILES"])

def _unimol_features(df: pl.DataFrame) -> np.ndarray:
    return _unimol.transform(df["SMILES"])

def _rdkit_features(df: pl.DataFrame) -> np.ndarray:
    return _rdkit.transform(df["SMILES"])


def _chemeleon_features(df: pl.DataFrame) -> np.ndarray:
    return _chemeleon.transform(df["SMILES"])


def _morgan_features(df: pl.DataFrame) -> np.ndarray:
    return _morgan.transform(df["SMILES"])


def _count_morgan_features(df: pl.DataFrame) -> np.ndarray:
    return _count_morgan.transform(df["SMILES"])


def _maccs_features(df: pl.DataFrame) -> np.ndarray:
    return _maccs.transform(df["SMILES"])


def _mordred_features(df: pl.DataFrame) -> np.ndarray:
    return _mordred.transform(df["SMILES"])


def _cyp_heme_features(df: pl.DataFrame) -> np.ndarray:
    return _cyp_heme.transform(df["SMILES"])


def _cyp_isoform_features(df: pl.DataFrame) -> np.ndarray:
    return _cyp_isoform.transform(df["SMILES"])


def _cyp_tdi_features(df: pl.DataFrame) -> np.ndarray:
    return _cyp_tdi.transform(df["SMILES"])


def _erg_features(df: pl.DataFrame) -> np.ndarray:
    return _erg.transform(df["SMILES"])


def _xtb_solv_features(df: pl.DataFrame) -> np.ndarray:
    return _xtb_solv.transform(df["SMILES"])


def _xtb_elec_features(df: pl.DataFrame) -> np.ndarray:
    return _xtb_elec.transform(df["SMILES"])


def _cyp_mech_features(df: pl.DataFrame) -> np.ndarray:
    return _cyp_mech.transform(df["SMILES"])


INPUT_REGISTRY: dict[str, Callable[[pl.DataFrame], np.ndarray]] = {
    _rdkit.name: _rdkit_features,
    _chemeleon.name: _chemeleon_features,
    _morgan.name: _morgan_features,
    _count_morgan.name: _count_morgan_features,
    _maccs.name: _maccs_features,
    _mordred.name: _mordred_features,
    _shape.name: _shape_features,
    _unimol.name: _unimol_features,
    # --- R2 pharmacophore blocks (append-only; versioned names pin the motif table) ---
    _cyp_heme.name: _cyp_heme_features,
    _cyp_isoform.name: _cyp_isoform_features,
    _cyp_tdi.name: _cyp_tdi_features,
    _erg.name: _erg_features,
    # --- GFN2-xTB blocks (tblite; conformer ensembles, slow: the cache is essential) ---
    _xtb_solv.name: _xtb_solv_features,
    _xtb_elec.name: _xtb_elec_features,
    # --- mechanistic block ported from Lizard Wizard Gizzard (see cyp_mech.py) ---
    _cyp_mech.name: _cyp_mech_features,
}


def is_known_input(name: str) -> bool:
    """True for a registered input or a `chemprop[/path/to/weights]` or `unimol[/path/to/model_0.pth]` spec."""
    return name in INPUT_REGISTRY or parse_spec(name) is not None or parse_unimol_spec(name) is not None


def _featurize_one(df: pl.DataFrame, input_name: str, cache_dir: Path) -> np.ndarray:
    """Return features for df, computing only molecules not already in the cache.

    Cache is content-addressed by SMILES, not by split/seed — a molecule's
    representation doesn't depend on how the data was split, so the same
    cache is reused across every split type and seed.

    Cache layout (one pair of files per representation):
      {input_name}.npy  — float64 array of shape (N_cached, D); grows as new molecules arrive
      {input_name}.idx  — text file, one SMILES per line; line i = row i of .npy
    """
    smiles_req = df["SMILES"].to_list()
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_spec(input_name)
    unimol_weights = parse_unimol_spec(input_name)
    if weights is not None:
        stem = cache_name(input_name)
    elif unimol_weights is not None:
        stem = unimol_cache_name(input_name)
    else:
        stem = input_name
    npy_path = cache_dir / f"{stem}.npy"
    idx_path = cache_dir / f"{stem}.idx"

    if idx_path.exists():
        indexed = idx_path.read_text().splitlines()
        pos: dict[str, int] = {s: i for i, s in enumerate(indexed)}
    else:
        indexed, pos = [], {}

    missing = list(dict.fromkeys(s for s in smiles_req if s not in pos))
    if missing:
        missing_df = pl.DataFrame({"SMILES": missing})
        if weights is not None:
            new_features = ChempropWeights(weights).transform(missing_df["SMILES"])
        elif unimol_weights is not None:
            new_features = UniMol(unimol_weights).transform(missing_df["SMILES"])
        else:
            new_features = INPUT_REGISTRY[input_name](missing_df)
        full = np.vstack([np.load(str(npy_path)), new_features]) if npy_path.exists() else new_features
        np.save(str(npy_path), full)
        for s in missing:
            pos[s] = len(indexed)
            indexed.append(s)
        idx_path.write_text("\n".join(indexed))
        print(f"[{input_name}] appended {len(missing)} molecules (cache now {len(indexed)} total)")
    else:
        print(f"[{input_name}] loaded from cache ({len(indexed)} molecules indexed)")

    return np.load(str(npy_path))[[pos[s] for s in smiles_req]]


def featurize(
    df: pl.DataFrame,
    input_names: str | list[str],
    cache_dir: Path = Path("data/features"),
) -> np.ndarray:
    """Return feature matrix for df, computing only uncached molecules.

    Pass a single name or a list; multiple inputs are hstacked after loading
    each from its own cache file.
    """
    names = [input_names] if isinstance(input_names, str) else input_names
    unknown = [n for n in names if not is_known_input(n)]
    if unknown:
        raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())} or chemprop[/path/to/weights] or unimol[/path/to/model_0.pth]")
    arrays = [_featurize_one(df, name, cache_dir) for name in names]
    return arrays[0] if len(arrays) == 1 else np.hstack(arrays)
