"""Evaluation-time importance weights, so a held-out slice imitates the blind set.

Why weights rather than a better split: the blind set sits in denser chemistry
than the training table does relative to itself. Only about 9% of training
molecules have a neighbour (outside their own scaffold family) as close as the
blind set's median, while half the blind set is above it. No subset of the
training data reproduces that regime, so `split_column`'s blind-matched strategy
can match scaffold novelty and not similarity. Re-weighting is not bounded that
way: a molecule can count for more than one molecule, which lets the scarce
dense rows carry the mass the blind set gives them.

What it costs is variance. Up-weighting a handful of rows shrinks the effective
sample size, and past some point the weighted metric is noisier than the bias it
removes. `effective_sample_size` is therefore reported with every weight vector,
and reference mass the slice cannot cover at all is reported as unsupported
rather than silently redistributed.

These are *evaluation* weights for distribution shift. They are unrelated to
training-time sample weights derived from label measurement error, which T7
owns in `weights.py`.
"""

import numpy as np

from data_tools.split_column import _morgan_fingerprints, _nearest_neighbour, _scaffold_of


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: how many equally-weighted molecules this is worth."""
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    return total**2 / float(np.sum(weights**2))


def weighted_rmse(errors: np.ndarray, weights: np.ndarray) -> float:
    """RMSE with each residual counted `weights` times."""
    errors = np.asarray(errors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(np.sqrt(float(np.sum(weights * errors**2)) / total))


def _cell_assignment(
    novel: np.ndarray, similarity: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Label each molecule with a (novelty, similarity-bin) cell index."""
    bin_index = np.digitize(similarity, edges, right=False)
    return np.asarray(novel, dtype=int) * (len(edges) + 1) + bin_index


def importance_weights(
    slice_novel: np.ndarray,
    slice_nn: np.ndarray,
    reference_novel: np.ndarray,
    reference_nn: np.ndarray,
    *,
    n_bins: int = 4,
    cap: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Weights making the slice's (novelty, similarity) mix match the reference's.

    Similarity bin edges come from the reference quantiles, so the reference
    spreads itself evenly across bins and the weights are driven by how the
    slice differs from it rather than by an arbitrary grid.
    """
    slice_nn = np.asarray(slice_nn, dtype=float)
    reference_nn = np.asarray(reference_nn, dtype=float)
    if len(slice_nn) == 0:
        return np.zeros(0), {
            "effective_sample_size": 0.0,
            "unsupported_reference_mass": 0.0,
            "max_weight": 0.0,
            "zero_weight_fraction": 0.0,
        }

    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(reference_nn, quantiles)) if n_bins > 1 else np.array([])
    slice_cells = _cell_assignment(slice_novel, slice_nn, edges)
    reference_cells = _cell_assignment(reference_novel, reference_nn, edges)

    weights = np.zeros(len(slice_nn), dtype=float)
    unsupported = 0.0
    for cell in np.unique(reference_cells):
        target = float(np.mean(reference_cells == cell))
        members = slice_cells == cell
        if not members.any():
            unsupported += target
            continue
        weights[members] = target / float(np.mean(members))

    if weights.sum() > 0:
        weights *= len(weights) / weights.sum()
    if cap is not None:
        weights = np.minimum(weights, cap)
        # Renormalising after a cap would reintroduce values above it, so the
        # mean is left below one and only the ratios between rows are used.

    return weights, {
        "effective_sample_size": effective_sample_size(weights),
        "unsupported_reference_mass": unsupported,
        "max_weight": float(weights.max()) if len(weights) else 0.0,
        "zero_weight_fraction": float(np.mean(weights == 0.0)),
    }


def shift_weights(
    slice_smiles: list[str],
    train_smiles: list[str],
    reference_smiles: list[str],
    *,
    n_bins: int = 4,
    cap: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Importance weights for a held-out slice, measured against a reference set.

    Both the slice and the reference are described by distance to the *same*
    training molecules, which is what the model actually saw, so the two
    distributions are comparable.
    """
    train_fps = _morgan_fingerprints(train_smiles)
    slice_nn = _nearest_neighbour(_morgan_fingerprints(slice_smiles), train_fps)
    reference_nn = _nearest_neighbour(_morgan_fingerprints(reference_smiles), train_fps)

    train_scaffolds = {_scaffold_of(smi, i) for i, smi in enumerate(train_smiles)}
    slice_novel = np.array(
        [_scaffold_of(smi, i) not in train_scaffolds for i, smi in enumerate(slice_smiles)]
    )
    reference_novel = np.array(
        [_scaffold_of(smi, i) not in train_scaffolds for i, smi in enumerate(reference_smiles)]
    )

    weights, report = importance_weights(
        slice_novel, slice_nn, reference_novel, reference_nn, n_bins=n_bins, cap=cap
    )
    positive = weights > 0
    report.update(
        {
            "slice_median_nn": float(np.median(slice_nn)),
            "reference_median_nn": float(np.median(reference_nn)),
            "weighted_median_nn": float(
                _weighted_median(slice_nn[positive], weights[positive])
            )
            if positive.any()
            else 0.0,
            "slice_novel_fraction": float(np.mean(slice_novel)),
            "reference_novel_fraction": float(np.mean(reference_novel)),
            "weighted_novel_fraction": float(
                np.sum(weights * slice_novel) / np.sum(weights)
            )
            if weights.sum() > 0
            else 0.0,
        }
    )
    return weights, report


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of `values` under `weights`."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return float(np.interp(0.5, cumulative, values))
