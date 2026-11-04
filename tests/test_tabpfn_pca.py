# tests/test_tabpfn_pca.py
"""TorchPCA agrees with sklearn's PCA up to component sign."""

import numpy as np
import pytest

pytest.importorskip("torch")

from sklearn.decomposition import PCA

from models.TabPFN import TorchPCA


def test_matches_sklearn_pca_up_to_sign():
    X = np.random.default_rng(0).normal(size=(200, 40))
    ours = TorchPCA(n_components=5, device="cpu").fit(X).transform(X)
    ref = PCA(n_components=5).fit_transform(X)
    np.testing.assert_allclose(np.abs(ours), np.abs(ref), atol=1e-3)


def test_output_shape_and_set_params():
    X = np.random.default_rng(1).normal(size=(50, 12))
    pca = TorchPCA(device="cpu")
    pca.set_params(n_components=3)
    assert pca.fit(X).transform(X).shape == (50, 3)
