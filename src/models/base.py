"""Abstract base class for CYP challenge models."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np



class CYPModel(ABC):
    """Base class for CYP challenge models.

    Operates on pre-computed feature matrices: fit(X, y) / predict(X).
    """

    name: ClassVar[str] = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on feature matrix X and target vector y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for every row in X."""
