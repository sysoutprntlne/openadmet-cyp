"""Shared preprocessing utilities for CYP models."""

import sys
from typing import Optional, Sequence, Tuple

import numpy as np


def drop_nan_rows(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    label: str = "",
    *,
    return_mask: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]] | Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Drop rows with any non-finite feature, or (if y given) a non-finite target.

    CYP target columns are sparse — most compounds were only assayed against
    a subset of the four enzymes — so a non-finite/missing y is expected and
    must be filtered here, not just non-finite X (invalid SMILES).
    Prints a message if any rows are dropped.
    Returns (X_clean, y_clean) — y_clean is None when y is None.
    With return_mask=True, also return the same row mask for aligned metadata.
    """
    mask = np.isfinite(X).all(axis=1)
    if y is not None:
        mask &= np.isfinite(y)
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        tag = f" ({label})" if label else ""
        print(f"Dropped {n_dropped} of {len(X)} examples with non-finite features/target{tag}")
    cleaned = (X[mask], y[mask] if y is not None else None)
    return (*cleaned, mask) if return_mask else cleaned


def zero_nonfinite_features(
    X: np.ndarray,
    label: str = "",
    names: Optional[Sequence[str]] = None,
    *,
    max_listed: int = 5,
) -> np.ndarray:
    """Return a copy of X with NaN/inf feature values set to 0, reporting where they were.

    Rows are kept (unlike drop_nan_rows) so a molecule whose featurizer failed still
    gets a prediction. Warns on stderr with the affected row indices (and names, e.g.
    SMILES, when given) and how many feature columns were zeroed in each.
    """
    bad = ~np.isfinite(X)
    bad_rows = np.flatnonzero(bad.any(axis=1))
    if len(bad_rows) == 0:
        return X
    tag = f" ({label})" if label else ""
    print(
        f"WARNING: zeroed {int(bad.sum())} non-finite feature values in {len(bad_rows)} of {len(X)} rows{tag}",
        file=sys.stderr,
    )
    for i in bad_rows[:max_listed]:
        who = f" {names[i]}" if names is not None else ""
        print(f"  row {i}{who}: {int(bad[i].sum())} of {X.shape[1]} features", file=sys.stderr)
    if len(bad_rows) > max_listed:
        print(f"  ... and {len(bad_rows) - max_listed} more rows", file=sys.stderr)
    X = X.copy()
    X[bad] = 0.0
    return X
