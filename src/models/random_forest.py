"""Random forest regressor for precomputed CYP features."""

from typing import ClassVar

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models.base import CYPModel


class RandomForest(CYPModel):
    """Bagged trees; callers filter missing labels with models.utils.drop_nan_rows."""

    name: ClassVar[str] = "random_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion="squared_error",
            bootstrap=True,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
