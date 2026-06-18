from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

from .base import SurrogateModel


@dataclass
class RBFConfig:
    kernel: str = "thin_plate_spline"
    epsilon: float | None = None
    neighbors: int | None = None


class _RBFRegressor(BaseEstimator, RegressorMixin):
    """
    Small sklearn-compatible wrapper around SciPy RBFInterpolator
    so it can be used inside a Pipeline.
    """

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        epsilon: float | None = None,
        neighbors: int | None = None,
    ):
        self.kernel = kernel
        self.epsilon = epsilon
        self.neighbors = neighbors
        self.model_: RBFInterpolator | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RBFRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        self.model_ = RBFInterpolator(
            X,
            y.reshape(-1, 1),
            kernel=self.kernel,
            epsilon=self.epsilon,
            neighbors=self.neighbors,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("_RBFRegressor is not fitted yet.")

        X = np.asarray(X, dtype=float)
        y_pred = self.model_(X).ravel()
        return np.asarray(y_pred, dtype=float)


class RBFSurrogate(SurrogateModel):
    name: str = "rbf"

    def __init__(self, config: Optional[RBFConfig] = None, **params: Any):
        super().__init__(**params)
        self.config = config or RBFConfig()
        self.model: Optional[Pipeline] = None

    @property
    def supports_uncertainty(self) -> bool:
        return False

    def _build_model(self) -> Pipeline:
        rbf = _RBFRegressor(
            kernel=self.config.kernel,
            epsilon=self.config.epsilon,
            neighbors=self.config.neighbors,
        )

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("rbf", rbf),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        self.model = self._build_model()
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RBFSurrogate is not fitted yet.")

        X = np.asarray(X, dtype=float)
        return np.asarray(self.model.predict(X)).ravel()

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        y_pred = self.predict(X)
        return y_pred, None

