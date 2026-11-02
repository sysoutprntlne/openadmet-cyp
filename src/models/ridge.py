"""Ridge regressor for the CYP challenge."""

from typing import ClassVar
                                                                                                
import numpy as np
from sklearn.linear_model import Ridge as RidgeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import CYPModel

DEFAULT_ALPHA = 1.0


class Ridge(CYPModel):
    """Standardized Ridge regression on finite, precomputed features and labels.

    Callers filter missing labels with models.utils.drop_nan_rows, as in the
    evaluator. Scaling is learned during fit and reused unchanged in predict.
    """

    name: ClassVar[str] = "ridge"

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self._model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeRegressor(alpha=alpha)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
    