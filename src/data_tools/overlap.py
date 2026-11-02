"""Measure structural overlap and duplication between datasets.

Raw SMILES comparison under-reports overlap: the same molecule arrives from
ChEMBL as a hydrochloride, from PubChem as a free base, and from the
challenge with explicit stereo. Everything here keys on the InChIKey
connectivity block (see data_tools.standardize) so those collapse together.

Two questions this answers:
  1. Duplication *within* a source -- do we have the same skeleton twice,
     with what label spread? (A wide spread means averaging is lossy.)
  2. Overlap *between* sources -- specifically, does any external source
     touch the blind test set? A non-zero train/test overlap is a scoring
     artefact we cannot fix, but a non-zero external/test overlap would mean
     our augmentation is leaking answers, which we can.
"""

import statistics
import sys

import polars as pl

from data_tools.standardize import connectivity_keys, stereo_keys

_KEY = "inchikey_block"
_STEREO_KEY = "inchikey_full"


def add_keys(df: pl.DataFrame, smiles_column: str = "SMILES") -> pl.DataFrame:
    """Return df with an inchikey_block column, dropping rows that fail to parse."""
    if _KEY in df.columns and df[_KEY].null_count() == 0:
        return df
    keys = connectivity_keys(df[smiles_column].to_list())
    out = df.with_columns(pl.Series(_KEY, keys))
    n_bad = out[_KEY].null_count()
    if n_bad:
        print(f"  dropped {n_bad} unparsable SMILES", file=sys.stderr)
    return out.filter(pl.col(_KEY).is_not_null())


def add_stereo_keys(df: pl.DataFrame, smiles_column: str = "SMILES") -> pl.DataFrame:
    """Return df with a full-InChIKey column (stereo-distinguishing)."""
    if _STEREO_KEY in df.columns and df[_STEREO_KEY].null_count() == 0:
        return df
    keys = stereo_keys(df[smiles_column].to_list())
    out = df.with_columns(pl.Series(_STEREO_KEY, keys))
    return out.filter(pl.col(_STEREO_KEY).is_not_null())


def duplication_report(df: pl.DataFrame, name: str, target_columns: list[str]) -> None:
    """Print how many connectivity blocks appear more than once, and label spread."""
    counts = df.group_by(_KEY).len()
    n_unique = len(counts)
    n_dup_keys = len(counts.filter(pl.col("len") > 1))
    print(f"\n{name}: {len(df)} rows -> {n_unique} unique skeletons ({n_dup_keys} appear >1x)")

    if not n_dup_keys:
        return
    dup_keys = set(counts.filter(pl.col("len") > 1)[_KEY].to_list())
    dups = df.filter(pl.col(_KEY).is_in(dup_keys))
    for target in target_columns:
        if target not in dups.columns:
            continue
        spreads: list[float] = []
        for (_key,), group in dups.group_by([_KEY], maintain_order=True):
            values = [v for v in group[target].to_list() if v is not None]
            if len(values) > 1:
                spreads.append(max(values) - min(values))
        if spreads:
            print(
                f"  {target}: {len(spreads)} duplicated skeletons with >1 label, "
                f"max spread {max(spreads):.3f}, median spread {statistics.median(spreads):.3f}"
            )


def overlap_report(left: pl.DataFrame, right: pl.DataFrame, left_name: str, right_name: str) -> int:
    """Print and return the number of connectivity blocks shared by two datasets."""
    left_keys = set(left[_KEY].to_list())
    right_keys = set(right[_KEY].to_list())
    shared = left_keys & right_keys
    pct_left = 100.0 * len(shared) / len(left_keys) if left_keys else 0.0
    pct_right = 100.0 * len(shared) / len(right_keys) if right_keys else 0.0
    print(
        f"{left_name} ({len(left_keys)}) n {right_name} ({len(right_keys)}) = "
        f"{len(shared)} shared  [{pct_left:.2f}% of {left_name}, {pct_right:.2f}% of {right_name}]"
    )
    return len(shared)


def collapse_duplicates(
    df: pl.DataFrame,
    target_columns: list[str],
    key: str = _KEY,
) -> pl.DataFrame:
    """Collapse rows sharing `key`, averaging numeric labels.

    One row per key, each label the mean of its measurements (nulls ignored);
    SMILES taken from the first occurrence.

    Choose `key` deliberately. Within a single source, pass _STEREO_KEY (full
    InChIKey): stereoisomers share a connectivity block but are different
    molecules with different labels, so collapsing them on connectivity
    would average real signal away. _KEY (connectivity block) is for joining
    *across* sources, where salt and stereo variants should match.
    """
    present = [t for t in target_columns if t in df.columns]
    aggs = [pl.col("SMILES").first()] + [pl.col(t).mean() for t in present]
    other = [c for c in df.columns if c not in present + ["SMILES", key]]
    aggs += [pl.col(c).first() for c in other]
    return df.group_by(key, maintain_order=True).agg(aggs).select([key, "SMILES"] + present + other)
