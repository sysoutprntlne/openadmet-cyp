"""Load local CYP challenge CSV splits into Polars DataFrames, with train/val splitting."""

import random
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from data_tools.split_column import _butina_groups, _scaffold_groups

_DEFAULT_DATA_DIR = Path("data")

# Label columns are sparse: in a multi-source union table a column can be null
# for the first thousands of rows, so polars' default 100-row schema inference
# types it String and every downstream numeric op fails. Always infer from the
# whole file.
_INFER_ALL = None

_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"
_TDI_TRAIN_FILE = "tdi_train.csv"
_EMAX_TRAIN_FILE = "emax_train.csv"
_SINGLE_CONC_TRAIN_FILE = "single_concentration_train.csv"


def load_train(path: Path = _DEFAULT_DATA_DIR / _TRAIN_FILE) -> pl.DataFrame:
    """Return the direct-inhibition training split as a Polars DataFrame."""
    return pl.read_csv(path, infer_schema_length=_INFER_ALL)


def load_test(path: Path = _DEFAULT_DATA_DIR / _TEST_FILE) -> pl.DataFrame:
    """Return the blinded test split as a Polars DataFrame."""
    return pl.read_csv(path, infer_schema_length=_INFER_ALL)


def load_tdi_train(path: Path = _DEFAULT_DATA_DIR / _TDI_TRAIN_FILE) -> pl.DataFrame:
    """Return the time-dependent inhibition (TDI) training split as a Polars DataFrame."""
    return pl.read_csv(path, infer_schema_length=_INFER_ALL)


def load_emax_train(path: Path = _DEFAULT_DATA_DIR / _EMAX_TRAIN_FILE) -> pl.DataFrame:
    """Return the Emax training split as a Polars DataFrame."""
    return pl.read_csv(path, infer_schema_length=_INFER_ALL)


def load_single_concentration_train(path: Path = _DEFAULT_DATA_DIR / _SINGLE_CONC_TRAIN_FILE) -> pl.DataFrame:
    """Return the single-concentration training split as a Polars DataFrame."""
    return pl.read_csv(path, infer_schema_length=_INFER_ALL)


def _random_split(df: pl.DataFrame, val_fraction: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Seeded random split of df into (train, val)."""
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def _scaffold_split(
    df: pl.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split df by Bemis-Murcko scaffold so no scaffold family spans both splits."""
    try:
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("rdkit is required for scaffold splitting") from exc

    scaffold_groups: dict[str, list[int]] = {}
    for idx, smi in enumerate(df["SMILES"].to_list()):
        try:
            scaffold = MurckoScaffoldSmiles(smi, includeChirality=False)
        except Exception:  # pylint: disable=broad-except
            scaffold = "__invalid__"
        scaffold_groups.setdefault(scaffold, []).append(idx)

    sorted_groups = sorted(scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True)
    random.Random(seed).shuffle(sorted_groups)

    n_total = len(df)
    val_indices: list[int] = []
    train_indices: list[int] = []
    for scaffold, indices in sorted_groups:
        if scaffold != "__invalid__" and (n_total == 0 or len(val_indices) / n_total < val_fraction):
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return df[train_indices], df[val_indices]


def _butina_split(
    df: pl.DataFrame,
    val_fraction: float,
    seed: int,
    cutoff: float = 0.4,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split df by Butina/Taylor clustering on Morgan fps; no similar molecules span both splits."""
    try:
        from rdkit import DataStructs  # type: ignore[import]
        from rdkit.Chem import MolFromSmiles  # type: ignore[import]
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore[import]
        from rdkit.ML.Cluster import Butina  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("rdkit is required for Butina splitting") from exc

    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    smiles = df["SMILES"].to_list()
    fps = []
    invalid: list[int] = []
    for idx, smi in enumerate(smiles):
        mol = MolFromSmiles(smi)
        if mol is None:
            invalid.append(idx)
            fps.append(None)
        else:
            fps.append(morgan_gen.GetFingerprint(mol))

    valid_fps = [fp for fp in fps if fp is not None]
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]

    dists: list[float] = []
    for i in range(1, len(valid_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend(1.0 - s for s in sims)

    raw_clusters = Butina.ClusterData(dists, len(valid_fps), cutoff, isDistData=True)
    clusters: list[list[int]] = [[valid_indices[j] for j in cluster] for cluster in raw_clusters]

    sorted_clusters = sorted(clusters, key=len, reverse=True)
    random.Random(seed).shuffle(sorted_clusters)

    n_total = len(df)
    val_indices: list[int] = []
    train_indices: list[int] = invalid[:]  # invalid SMILES always go to train
    for cluster in sorted_clusters:
        if n_total == 0 or len(val_indices) / n_total < val_fraction:
            val_indices.extend(cluster)
        else:
            train_indices.extend(cluster)

    return df[train_indices], df[val_indices]


def load_data(
    *,
    split_type: Literal["random", "scaffold", "butina"] = "random",
    val_fraction: float = 0.2,
    seed: int = 42,
    butina_cutoff: float = 0.4,
    data_dir: Path = Path("data"),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train.csv and return (train_df, val_df) split by split_type.

    split_type options:
      'random'   — seeded random split (default)
      'scaffold' — Bemis-Murcko scaffold split; no scaffold leakage
      'butina'   — Butina/Taylor cluster split on Morgan fps; no similar molecules span both splits
    """
    df = load_train(data_dir / _TRAIN_FILE)

    if split_type == "scaffold":
        return _scaffold_split(df, val_fraction, seed)
    if split_type == "butina":
        return _butina_split(df, val_fraction, seed, butina_cutoff)
    return _random_split(df, val_fraction, seed)


def cv_folds(
    df: pl.DataFrame,
    *,
    n_folds: int,
    split_type: Literal["random", "scaffold", "butina"] = "random",
    seed: int = 42,
    butina_cutoff: float = 0.4,
) -> np.ndarray:
    """Return a fold index in [0, n_folds) for each row of df.

    'scaffold' and 'butina' keep each scaffold family / cluster inside one fold;
    'random' assigns rows independently. Groups are placed largest-first into the
    currently smallest fold, so fold sizes stay balanced.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    smiles = df["SMILES"].to_list()
    if split_type == "scaffold":
        groups = list(_scaffold_groups(smiles).values())
    elif split_type == "butina":
        groups = list(_butina_groups(smiles, butina_cutoff).values())
    else:
        groups = [[idx] for idx in range(len(smiles))]

    if len(groups) < n_folds:
        raise ValueError(f"Only {len(groups)} {split_type} groups for {n_folds} folds")

    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)

    fold_sizes = [0] * n_folds
    folds = np.empty(len(smiles), dtype=int)
    for rows in groups:
        fold = min(range(n_folds), key=fold_sizes.__getitem__)
        folds[rows] = fold
        fold_sizes[fold] += len(rows)
    return folds
