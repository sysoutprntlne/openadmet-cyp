"""Offline tests for the TabPFN wrapper using synthetic numerical inputs."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from models.TabPFN import TabPFN


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
