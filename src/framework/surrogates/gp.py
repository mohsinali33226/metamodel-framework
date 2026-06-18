from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from .base import SurrogateModel


@dataclass
class GPConfig:
    normalize_y: bool = True
    random_state: int = 42
    n_restarts_optimizer: int = 3

        # Kernel settings
    length_scale_init: float = 1.0
    length_scale_bounds: Tuple[float, float] = (1e-4, 1e5)


    noise_level_init: float = 1e-6
    noise_level_bounds: Tuple[float, float] = (1e-12, 1.0)




class GPSurrogate(SurrogateModel):
    name: str = "gp"

    def __init__(self, config: Optional[GPConfig] = None, **params: Any):
        super().__init__(**params)
        self.config = config or GPConfig()
        self.model: Optional[Pipeline] = None

    @property
    def supports_uncertainty(self) -> bool:
        return True

    def _build_model(self, X: np.ndarray) -> Pipeline:
        d = int(X.shape[1])

        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(
                length_scale=np.ones(d) * self.config.length_scale_init,
                length_scale_bounds=self.config.length_scale_bounds,
            )
            + WhiteKernel(
                noise_level=self.config.noise_level_init,
                noise_level_bounds=self.config.noise_level_bounds,
            )
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=self.config.normalize_y,
            random_state=self.config.random_state,
            n_restarts_optimizer=self.config.n_restarts_optimizer,
        )

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("gp", gp),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        self.model = self._build_model(X)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GPSurrogate is not fitted yet.")
        X = np.asarray(X)
        return self.model.predict(X).ravel()

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.model is None:
            raise RuntimeError("GPSurrogate is not fitted yet.")
        X = np.asarray(X)

        # We need to call predict on the GP step with return_std=True
        scaler: StandardScaler = self.model.named_steps["scaler"]
        gp: GaussianProcessRegressor = self.model.named_steps["gp"]

        Xs = scaler.transform(X)
        y_mean, y_std = gp.predict(Xs, return_std=True)

        return np.asarray(y_mean).ravel(), np.asarray(y_std).ravel()


