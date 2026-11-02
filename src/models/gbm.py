"""LightGBM regressor for precomputed CYP features."""

from typing import ClassVar

import numpy as np

from models.base import CYPModel


class LightGBM(CYPModel):
    """Boosted trees; callers filter missing labels with models.utils.drop_nan_rows."""

    name: ClassVar[str] = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        from lightgbm import LGBMRegressor

        self._model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            boosting_type="gbdt",
            objective="regression",
            device_type="cpu",
            verbosity=-1,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
