from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .base import SurrogateModel


@dataclass
class SVRConfig:
    kernel: str = "rbf"
    C: float = 100.0
    epsilon: float = 0.001
    gamma: Union[str, float] = "scale"



class SVRSurrogate(SurrogateModel):
    name: str = "svr"

    def __init__(self, config: Optional[SVRConfig] = None, **params: Any):
        super().__init__(**params)
        self.config = config or SVRConfig()
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        svr = SVR(
            kernel=self.config.kernel,
            C=self.config.C,
            epsilon=self.config.epsilon,
            gamma=self.config.gamma,
        )

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", svr),
        ])

        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.asarray(self.model.predict(X)).ravel()

    def predict_with_uncertainty(self, X):
        y = self.predict(X)
        return y, None



