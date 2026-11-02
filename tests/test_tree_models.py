"""Synthetic contract checks for real T3 tree estimators, not CYP accuracy tests."""

import numpy as np
import pytest

from models import CYPModel, REGISTRY
from models.utils import drop_nan_rows


@pytest.mark.parametrize("name", ["random_forest", "lightgbm", "xgboost"])
@pytest.mark.parametrize("configured", [False, True], ids=["defaults", "configured"])
def test_real_estimator_fit_predict(name, configured):
    module, estimator_name = {
        "random_forest": ("sklearn.ensemble", "RandomForestRegressor"),
        "lightgbm": ("lightgbm", "LGBMRegressor"),
        "xgboost": ("xgboost", "XGBRegressor"),
    }[name]
    backend = pytest.importorskip(module)
    settings = {"n_estimators": 12, "random_state": 17, "n_jobs": 1} if configured else {}
    model = REGISTRY[name](**settings)
    assert isinstance(model, CYPModel)
    assert isinstance(model._model, getattr(backend, estimator_name))
    for key, value in settings.items():
        assert model._model.get_params()[key] == value
    rng = np.random.default_rng(2026)
    X_train = rng.normal(size=(64, 6))
    y_train = 2 * X_train[:, 0] - X_train[:, 1] + rng.normal(scale=0.1, size=64)
    y_train[3] = np.nan
    X_train, y_train = drop_nan_rows(X_train, y_train)
    X_val = rng.normal(size=(5, 6))

    assert model.fit(X_train, y_train) is None
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
    np.testing.assert_allclose(np.concatenate(singles), batch, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(reordered, batch[permutation], rtol=1e-6, atol=1e-8)


def test_registry_preserves_existing_order_and_appends_ensembles():
    assert list(REGISTRY) == ["decision_tree", "ridge", "random_forest", "lightgbm", "xgboost"]
    for name in ["random_forest", "lightgbm", "xgboost"]:
        assert REGISTRY[name].name == name
        assert issubclass(REGISTRY[name], CYPModel)
