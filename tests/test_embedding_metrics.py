"""Embedding metrics behave as documented on data where the answer is known."""

import numpy as np
import pytest

from representations import embedding_metrics as em


def _blobs(n: int, d: int, shift: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d)) + shift


def test_effective_rank_of_isotropic_data_is_near_dimension():
    X = np.random.default_rng(0).normal(size=(500, 20))
    assert em.effective_rank(X) > 17


def test_effective_rank_detects_collapse_onto_a_line():
    rng = np.random.default_rng(0)
    X = np.outer(rng.normal(size=500), rng.normal(size=20)) + 1e-3 * rng.normal(size=(500, 20))
    assert em.effective_rank(X) < 1.5


def test_mean_pair_difference_matches_brute_force():
    y = np.random.default_rng(1).normal(size=40)
    brute = np.mean([abs(y[i] - y[j]) for i in range(40) for j in range(i + 1, 40)])
    assert em._mean_pair_difference(y) == pytest.approx(brute)


def test_roughness_is_low_when_labels_follow_the_embedding_and_near_one_when_shuffled():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = X[:, 0]
    assert em.roughness(X, y) < 0.4
    assert em.roughness(X, rng.permutation(y)) == pytest.approx(1.0, abs=0.2)


def test_roughness_handles_exact_duplicates():
    X = np.vstack([np.zeros((3, 2)), np.ones((3, 2))])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert em.roughness(X, y) == pytest.approx(0.0)


def test_shift_stats_match_for_same_distribution_and_flag_a_shifted_set():
    train, val_same, val_far = _blobs(600, 10, 0, 0), _blobs(300, 10, 0, 1), _blobs(300, 10, 4, 2)
    _, d_train = em.nn_distances(train, val_same)
    same = em.shift_stats(em.nn_distances(train, val_same)[0], d_train)
    far = em.shift_stats(em.nn_distances(train, val_far)[0], d_train)
    assert same["nn_ratio"] == pytest.approx(1.0, abs=0.15)
    assert same["frac_out"] < 0.15
    assert far["nn_ratio"] > 1.5
    assert far["frac_out"] > 0.9


def test_two_sample_auc_near_half_for_same_distribution_and_high_when_shifted():
    train = _blobs(400, 10, 0, 0)
    assert em.two_sample_auc(train, _blobs(200, 10, 0, 1)) == pytest.approx(0.5, abs=0.1)
    assert em.two_sample_auc(train, _blobs(200, 10, 3, 2)) > 0.95


def test_mmd_p_value_separates_same_from_shifted():
    train = _blobs(300, 10, 0, 0)
    _, p_same = em.mmd_rbf(train, _blobs(300, 10, 0, 1))
    _, p_far = em.mmd_rbf(train, _blobs(300, 10, 1, 2))
    assert p_same > 0.05
    assert p_far < 0.05


def test_tanimoto_nn_distance_is_zero_for_a_copy_and_leave_one_out_skips_self():
    rng = np.random.default_rng(0)
    train = (rng.random((50, 128)) < 0.2).astype(np.uint8)
    d_val, d_train = em.tanimoto_nn_distances(train, train[:5])
    assert np.allclose(d_val, 0.0)
    assert (d_train > 0).all()


def test_neighbour_recall_is_perfect_for_the_fingerprint_itself():
    rng = np.random.default_rng(0)
    fp = (rng.random((120, 256)) < 0.15).astype(np.uint8)
    # a 1:1 monotone map of the bits keeps Euclidean order == Hamming order, not Tanimoto, so only
    # require it to beat a random embedding by a wide margin
    informed = em.neighbour_recall(fp.astype(float), fp, k=5)
    random_emb = em.neighbour_recall(rng.normal(size=(120, 16)), fp, k=5)
    assert informed > random_emb + 0.3


def test_evaluate_embedding_reports_every_metric_and_ranks_an_informative_embedding_higher():
    rng = np.random.default_rng(0)
    n = 400
    latent = rng.normal(size=(n, 4))
    y = latent[:, 0] + 0.1 * rng.normal(size=n)
    fp = (rng.random((n, 128)) < 0.2).astype(np.uint8)
    split = 300
    informative = np.hstack([latent, 0.05 * rng.normal(size=(n, 4))])
    noise = rng.normal(size=(n, 8))

    def run(X):
        return em.evaluate_embedding(
            X[:split], X[split:], {"t": y[:split]}, {"t": y[split:]}, fp[:split], fp[split:], max_samples=300, seed=0,
        )

    good, bad = run(informative), run(noise)
    expected = {
        "dim", "nn_ratio", "frac_out", "eff_rank", "recall@10", "two_sample_auc", "mmd", "mmd_p",
        "probe_r2", "knn_r2", "roughness", "err_dist_rho",
    }
    assert expected <= set(good)
    assert good["probe_r2"] > 0.8 > bad["probe_r2"]
    assert good["roughness"] < bad["roughness"]


def test_evaluate_embedding_skips_a_target_with_too_few_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    fp = (rng.random((120, 64)) < 0.2).astype(np.uint8)
    sparse = np.full(120, np.nan)
    sparse[:3] = 1.0
    out = em.evaluate_embedding(
        X[:90], X[90:], {"t": sparse[:90]}, {"t": sparse[90:]}, fp[:90], fp[90:], max_samples=100, seed=0,
    )
    assert np.isnan(out["probe_r2"])


def test_evaluate_shift_needs_no_labels_and_flags_a_shifted_set():
    train = _blobs(500, 10, 0, 0)
    same = em.evaluate_shift(train, _blobs(250, 10, 0, 1), max_samples=300, seed=0)
    far = em.evaluate_shift(train, _blobs(250, 10, 3, 2), max_samples=300, seed=0)
    assert {"nn_ratio", "frac_out", "two_sample_auc", "mmd_p"} <= set(same)
    assert same["two_sample_auc"] < 0.65 < 0.95 < far["two_sample_auc"]
    assert far["nn_ratio"] > same["nn_ratio"]
    assert far["mmd_p"] < 0.05
