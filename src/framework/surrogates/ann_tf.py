from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from .base import SurrogateModel


@dataclass
class ANNConfig:
    hidden_units_1: int = 64
    hidden_units_2: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.1
    verbose: int = 0
    random_state: int = 42


class ANNTFSurrogate(SurrogateModel):
    name: str = "ann_tf"

    def __init__(self, config: Optional[ANNConfig] = None, **params: Any):
        super().__init__(**params)
        self.config = config or ANNConfig()
        self.x_scaler: Optional[StandardScaler] = None
        self.y_scaler: Optional[StandardScaler] = None
        self.model: Optional[tf.keras.Model] = None

    @property
    def supports_uncertainty(self) -> bool:
        return False

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        tf.keras.utils.set_random_seed(self.config.random_state)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(self.config.hidden_units_1, activation="relu"),
            tf.keras.layers.Dense(self.config.hidden_units_2, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
        )
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANNTFSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel().reshape(-1, 1)

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        Xs = self.x_scaler.fit_transform(X)
        ys = self.y_scaler.fit_transform(y)

        self.model = self._build_model(input_dim=X.shape[1])
        self.model.fit(
            Xs,
            ys,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            verbose=self.config.verbose,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.x_scaler is None or self.y_scaler is None:
            raise RuntimeError("ANNTFSurrogate is not fitted yet.")

        X = np.asarray(X, dtype=float)
        Xs = self.x_scaler.transform(X)
        y_scaled = self.model.predict(Xs, verbose=0)
        y = self.y_scaler.inverse_transform(y_scaled)
        return np.asarray(y).ravel()

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        y = self.predict(X)
        return y, None

