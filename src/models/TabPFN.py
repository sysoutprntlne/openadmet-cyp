"""TabPFN regressor for the CYP challenge.

TabPFN-3.5 accepts at most 20,000 features and 1,000,000 rows, so wide
representations may need reducing before they fit.
"""

from typing import ClassVar, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from models.base import CYPModel


class TorchPCA(TransformerMixin, BaseEstimator):
    """Exact PCA via torch SVD; runs on the GPU when one is available, else CPU."""

    def __init__(self, n_components: int = 256, device: str = "auto") -> None:
        self.n_components = n_components
        self.device = device

    def _device(self):
        import torch

        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y=None) -> "TorchPCA":
        import torch

        x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self._device())
        self.mean_ = x.mean(dim=0)
        _, _, vt = torch.linalg.svd(x - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        import torch

        x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.mean_.device)
        return ((x - self.mean_) @ self.components_.T).cpu().numpy()


class TabPFN(CYPModel):
    """In-context tabular regressor; no gradient training happens in fit().

    Callers filter missing labels with models.utils.drop_nan_rows.
    """

    name: ClassVar[str] = "tabpfn"

    def __init__(
        self,
        n_estimators: int | Literal["auto"] = "auto",
        softmax_temperature: float | Literal["auto"] = "auto",
        ignore_pretraining_limits: bool = False,
        device: str = "auto",
        model_path: str = "auto",
        random_state: int = 42,
        n_components: int | None = 256,
    ) -> None:
        from tabpfn import TabPFNRegressor

        # Clamped to the data shape in fit(); None skips the PCA step. PCA uses the
        # same device as TabPFN.
        self.n_components = n_components
        self._model = Pipeline(
            [
                *([("pca", TorchPCA(device=device))] if n_components else []),
                (
                    "regressor",
                    TabPFNRegressor(
                        n_estimators=n_estimators,
                        softmax_temperature=softmax_temperature,
                        ignore_pretraining_limits=ignore_pretraining_limits,
                        device=device,
                        model_path=model_path,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.n_components:
            self._model.set_params(pca__n_components=min(self.n_components, *X.shape))
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
