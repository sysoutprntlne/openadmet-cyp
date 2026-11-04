# tests/test_shift_weights.py
"""Importance weights that make a held-out slice imitate the blind set's regime."""

import numpy as np
import pytest

from data_tools.shift_weights import (
    effective_sample_size,
    importance_weights,
    shift_weights,
    weighted_rmse,
)

BENZENES = ["Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "COc1ccccc1", "Clc1ccccc1"]
PYRIDINES = ["Cc1ccncc1", "CCc1ccncc1", "CCCc1ccncc1", "COc1ccncc1"]
ALIPHATICS = ["CCCCCCO", "CCCCCCCN", "CCCCCC(=O)O"]


def test_no_shift_gives_uniform_weights():
    nn = np.array([0.2, 0.4, 0.6, 0.8])
    novel = np.array([True, True, False, False])
    weights, report = importance_weights(novel, nn, novel, nn, n_bins=2)
    assert weights == pytest.approx(np.ones(4))
    assert report["unsupported_reference_mass"] == pytest.approx(0.0)


def test_weights_average_to_one():
    rng = np.random.default_rng(0)
    slice_nn, reference_nn = rng.uniform(size=40), rng.uniform(size=25)
    slice_novel = rng.integers(0, 2, 40).astype(bool)
    reference_novel = rng.integers(0, 2, 25).astype(bool)
    weights, _ = importance_weights(slice_novel, slice_nn, reference_novel, reference_nn)
    assert weights.mean() == pytest.approx(1.0)


def test_overrepresented_molecules_are_down_weighted():
    """The slice is mostly far molecules; the reference is mostly near ones."""
    slice_nn = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
    reference_nn = np.array([0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    novel_slice = np.zeros(8, dtype=bool)
    novel_reference = np.zeros(6, dtype=bool)
    weights, _ = importance_weights(novel_slice, slice_nn, novel_reference, reference_nn, n_bins=2)
    far, near = weights[slice_nn < 0.5], weights[slice_nn > 0.5]
    assert far.max() < near.min()
    assert np.allclose(far, far[0]) and np.allclose(near, near[0])


def test_novelty_is_matched_separately_from_similarity():
    """Two molecules at the same similarity can still be weighted differently."""
    slice_nn = np.array([0.5, 0.5, 0.5, 0.5])
    slice_novel = np.array([True, True, False, False])
    reference_nn = np.array([0.5, 0.5, 0.5, 0.5])
    reference_novel = np.array([True, True, True, False])
    weights, _ = importance_weights(slice_novel, slice_nn, reference_novel, reference_nn, n_bins=1)
    assert weights[slice_novel].mean() > weights[~slice_novel].mean()


def test_reference_mass_without_support_is_reported():
    """Reference regions the slice cannot reach are the honest failure mode."""
    slice_nn = np.array([0.1, 0.2, 0.15, 0.25])
    reference_nn = np.array([0.1, 0.95, 0.95, 0.95])
    novel_slice = np.zeros(4, dtype=bool)
    novel_reference = np.zeros(4, dtype=bool)
    _, report = importance_weights(
        novel_slice, slice_nn, novel_reference, reference_nn, n_bins=2
    )
    assert report["unsupported_reference_mass"] == pytest.approx(0.75)


def test_effective_sample_size_falls_as_weights_concentrate():
    uniform = np.ones(100)
    concentrated = np.append(np.full(99, 0.01), 99.01)
    assert effective_sample_size(uniform) == pytest.approx(100.0)
    assert effective_sample_size(concentrated) < 5.0


def test_capping_limits_the_largest_weight():
    slice_nn = np.append(np.full(39, 0.1), 0.9)
    reference_nn = np.append(np.full(39, 0.9), 0.1)
    novel_slice, novel_reference = np.zeros(40, dtype=bool), np.zeros(40, dtype=bool)
    uncapped, _ = importance_weights(
        novel_slice, slice_nn, novel_reference, reference_nn, n_bins=2
    )
    capped, report = importance_weights(
        novel_slice, slice_nn, novel_reference, reference_nn, n_bins=2, cap=5.0
    )
    assert uncapped.max() > 5.0
    assert capped.max() <= 5.0 + 1e-9
    assert report["effective_sample_size"] > effective_sample_size(uncapped)


def test_weighted_rmse_matches_plain_rmse_when_uniform():
    errors = np.array([1.0, -2.0, 0.5, 0.0])
    assert weighted_rmse(errors, np.ones(4)) == pytest.approx(
        float(np.sqrt(np.mean(errors**2)))
    )


def test_weighted_rmse_follows_the_up_weighted_molecules():
    errors = np.array([0.1, 0.1, 2.0, 2.0])
    toward_accurate = weighted_rmse(errors, np.array([1.9, 1.9, 0.1, 0.1]))
    toward_inaccurate = weighted_rmse(errors, np.array([0.1, 0.1, 1.9, 1.9]))
    assert toward_accurate < toward_inaccurate


def test_shift_weights_reads_chemistry():
    """The wrapper turns SMILES into the two axes and reports the achieved match."""
    train = BENZENES + ALIPHATICS
    held_out = PYRIDINES + ALIPHATICS[:1]
    weights, report = shift_weights(held_out, train, BENZENES[:3] + PYRIDINES[:1], n_bins=2)
    assert len(weights) == len(held_out)
    assert weights.mean() == pytest.approx(1.0)
    assert 0.0 < report["effective_sample_size"] <= len(held_out)
    for key in ("slice_median_nn", "reference_median_nn", "weighted_median_nn"):
        assert 0.0 <= report[key] <= 1.0


def test_shift_weights_pulls_the_median_toward_the_reference():
    train = BENZENES + ALIPHATICS
    held_out = BENZENES[:2] + ALIPHATICS
    # A reference made of benzenes should pull weight onto the benzene-like rows.
    _, report = shift_weights(held_out, train, BENZENES, n_bins=2)
    toward = abs(report["weighted_median_nn"] - report["reference_median_nn"])
    away = abs(report["slice_median_nn"] - report["reference_median_nn"])
    assert toward <= away
