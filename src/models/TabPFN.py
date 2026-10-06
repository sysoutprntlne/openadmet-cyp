"""PCA-compressed TabPFN regressor for precomputed molecular features."""

from typing import ClassVar

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from models.base import CYPModel


class TabPFN(CYPModel):
    """Use labeled examples as context for a fixed pretrained TabPFN model.

    Callers supply finite features and labels via models.utils.drop_nan_rows.
    PCA is learned during fit and reused unchanged during predict.
    """

    name: ClassVar[str] = "tabpfn"

    def __init__(
        self,
        n_components: int = 200,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")

        # Import only when selected, so other models do not load TabPFN/PyTorch.
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        self._model = Pipeline(
            [
                ("pca", PCA(n_components=n_components, svd_solver="full")),
                (
                    "regressor",
                    TabPFNRegressor.create_default_for_version(
                        ModelVersion.V3_5, random_state=random_state, device=device
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit PCA and prepare TabPFN context using training data only."""
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix (rows, features).")
        n_components = self._model.named_steps["pca"].n_components
        if n_components > min(X.shape):
            raise ValueError(
                f"PCA requested {n_components} components, but X has {X.shape[0]} training rows "
                f"and {X.shape[1]} features. Use n_components <= {min(X.shape)}."
            )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform features with fitted PCA and return one prediction per row."""
        return self._model.predict(X)
