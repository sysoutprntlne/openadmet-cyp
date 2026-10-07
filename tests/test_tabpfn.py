"""Offline wrapper/integration checks; no real TabPFN inference or CYP accuracy claims."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from models.TabPFN import TabPFN
from models.utils import drop_nan_rows


@pytest.fixture
def tabpfn_backend(monkeypatch):
    """Replace the optional backend so no model weights can be loaded."""
    regressor = Mock()
    regressor.fit.side_effect = AssertionError("Invalid inputs must fail before backend fitting")
    factory = Mock(return_value=regressor)
    backend = ModuleType("tabpfn")
    backend.TabPFNRegressor = SimpleNamespace(create_default_for_version=factory)
    constants = ModuleType("tabpfn.constants")
    constants.ModelVersion = SimpleNamespace(V3_5="test-version")
    monkeypatch.setitem(sys.modules, "tabpfn", backend)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    return factory, regressor


@pytest.mark.parametrize("n_components", [0, -1, True, False, 1.5, "2", None])
def test_invalid_component_count(n_components, tabpfn_backend):
    factory, _ = tabpfn_backend

    with pytest.raises(ValueError, match=r"^n_components must be a positive integer\.$"):
        TabPFN(n_components=n_components)

    factory.assert_not_called()


@pytest.mark.parametrize("shape", [(3, 5), (5, 3)], ids=["too-few-rows", "too-few-features"])
def test_component_count_exceeds_training_capacity(shape, tabpfn_backend):
    _, regressor = tabpfn_backend
    model = TabPFN(n_components=4)
    X = np.arange(np.prod(shape), dtype=float).reshape(shape)
    y = np.arange(shape[0], dtype=float)

    with pytest.raises(ValueError) as error:
        model.fit(X, y)

    assert str(error.value) == (
        f"PCA requested 4 components, but X has {shape[0]} training rows "
        f"and {shape[1]} features. Use n_components <= 3."
    )
    regressor.fit.assert_not_called()


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Record PCA outputs and return a deterministic, row-wise numerical signal."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RecordingRegressor":
        self.X_fit_ = X.copy()
        self.y_fit_ = y.copy()
        self.weights_ = np.arange(1, X.shape[1] + 1, dtype=float)
        self.predict_inputs_ = []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "weights_")
        self.predict_inputs_.append(X.copy())
        return X @ self.weights_ + self.y_fit_.mean()


@pytest.fixture
def recording_backend(tabpfn_backend):
    """Patch the wrapper's lazy import/factory boundary, leaving sklearn intact."""
    factory, _ = tabpfn_backend
    regressor = RecordingRegressor()
    factory.return_value = regressor
    return regressor


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(2026)
    X_train = rng.normal(size=(12, 6)) + np.arange(6)
    y_train = rng.normal(size=12)
    X_val = rng.normal(size=(5, 6)) + np.arange(6) + 50.0
    return X_train, y_train, X_val


def test_fit_compresses_features_and_preserves_clean_label_alignment(recording_backend, synthetic_data):
    X_train, y_train, _ = synthetic_data
    y_train[1] = np.nan
    X_train[7, 2] = np.inf
    kept_rows = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    X_clean, y_clean = drop_nan_rows(X_train, y_train)
    model = TabPFN(n_components=3)

    model.fit(X_clean, y_clean)

    pca = model._model.named_steps["pca"]
    assert recording_backend.X_fit_.shape == (len(kept_rows), 3)
    assert np.isfinite(recording_backend.X_fit_).all()
    np.testing.assert_allclose(recording_backend.y_fit_, y_train[kept_rows])
    np.testing.assert_allclose(pca.mean_, X_train[kept_rows].mean(axis=0))
    expected = (X_train[kept_rows] - pca.mean_) @ pca.components_.T
    np.testing.assert_allclose(recording_backend.X_fit_, expected, rtol=1e-10, atol=1e-12)


def test_shifted_prediction_reuses_training_pca(monkeypatch, recording_backend, synthetic_data):
    X_train, y_train, X_val = synthetic_data
    model = TabPFN(n_components=3)
    model.fit(X_train, y_train)
    pca = model._model.named_steps["pca"]
    mean_before = pca.mean_.copy()
    components_before = pca.components_.copy()
    np.testing.assert_allclose(mean_before, X_train.mean(axis=0))
    assert not np.allclose(mean_before, np.vstack([X_train, X_val]).mean(axis=0))
    fit = Mock(side_effect=AssertionError("Prediction must not fit PCA"))
    fit_transform = Mock(side_effect=AssertionError("Prediction must not fit_transform PCA"))
    monkeypatch.setattr(pca, "fit", fit)
    monkeypatch.setattr(pca, "fit_transform", fit_transform)

    predictions = model.predict(X_val)

    fit.assert_not_called()
    fit_transform.assert_not_called()
    np.testing.assert_allclose(pca.mean_, mean_before, rtol=0, atol=0)
    np.testing.assert_allclose(pca.components_, components_before, rtol=0, atol=0)
    expected_features = (X_val - mean_before) @ components_before.T
    np.testing.assert_allclose(recording_backend.predict_inputs_[-1], expected_features, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(predictions, expected_features @ recording_backend.weights_ + y_train.mean())


def test_single_and_batch_predictions_preserve_row_order(recording_backend, synthetic_data):
    X_train, y_train, X_val = synthetic_data
    model = TabPFN(n_components=3)
    model.fit(X_train, y_train)

    batch = model.predict(X_val)
    singles = [model.predict(row[None, :]) for row in X_val]
    permutation = np.array([3, 0, 4, 1, 2])
    reordered = model.predict(X_val[permutation])

    assert batch.shape == (len(X_val),)
    assert np.isfinite(batch).all()
    for single in singles:
        assert single.shape == (1,)
        assert np.isfinite(single).all()
    assert reordered.shape == batch.shape
    assert np.isfinite(reordered).all()
    assert np.ptp(batch) > 1e-8  # A constant backend would make the order check vacuous.
    np.testing.assert_allclose(np.concatenate(singles), batch, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reordered, batch[permutation], rtol=1e-10, atol=1e-12)


def test_predict_before_fit_raises(recording_backend, synthetic_data):
    _, _, X_val = synthetic_data
    model = TabPFN(n_components=3)

    with pytest.raises(NotFittedError):
        model.predict(X_val)


def test_registry_model_obeys_evaluator_contract(recording_backend, synthetic_data):
    from models import REGISTRY
    from models.evaluate import _evaluate_model, _resolve_models

    X_train, y_train, X_val = synthetic_data
    y_val = np.random.default_rng(17).normal(size=len(X_val))
    (model_class,) = _resolve_models(["tabpfn"])
    assert model_class is REGISTRY["tabpfn"] is TabPFN
    model = model_class(n_components=3)

    metrics = _evaluate_model(model, X_train, y_train, X_val, y_val)

    assert len(recording_backend.predict_inputs_) == 1
    assert recording_backend.predict_inputs_[0].shape == (len(X_val), 3)
    predictions = recording_backend.predict_inputs_[0] @ recording_backend.weights_ + y_train.mean()
    residuals = y_val - predictions
    expected = {
        "RMSE": np.sqrt(np.mean(residuals**2)),
        "MAE": np.mean(np.abs(residuals)),
        "R2": 1 - np.sum(residuals**2) / np.sum((y_val - y_val.mean())**2),
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics == pytest.approx(expected)
