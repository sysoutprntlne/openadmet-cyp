"""Decision tree regressor for the CYP challenge."""

from typing import ClassVar, Optional

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from models.base import CYPModel

DEFAULT_MAX_DEPTH = 5
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_RANDOM_STATE = 42


class DecisionTree(CYPModel):
    """Single decision tree wrapping sklearn DecisionTreeRegressor."""

    name: ClassVar[str] = "decision_tree"

    def __init__(
        self,
        max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
        min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self._model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


