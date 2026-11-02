"""XGBoost regressor for precomputed CYP features."""

from typing import ClassVar

import numpy as np

from models.base import CYPModel


class XGBoost(CYPModel):
    """Boosted trees; callers filter missing labels with models.utils.drop_nan_rows."""

    name: ClassVar[str] = "xgboost"

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        from xgboost import XGBRegressor

        self._model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective="reg:squarederror",
            booster="gbtree",
            tree_method="hist",
            device="cpu",
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
