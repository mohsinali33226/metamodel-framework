from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from .base import SurrogateModel


@dataclass
class EnsembleConfig:
    members: list[SurrogateModel] = field(default_factory=list)
    validation_fraction: float = 0.2
    random_state: int = 42
    weight_epsilon: float = 1e-12


class EnsembleSurrogate(SurrogateModel):
    name: str = "ensemble"

    def __init__(self, config: Optional[EnsembleConfig] = None, **params: Any):
        super().__init__(**params)
        self.config = config or EnsembleConfig()

        # prototypes from config
        self.members: list[SurrogateModel] = self.config.members

        # fitted final members
        self.trained_members_: list[SurrogateModel] = []

        # learned validation-based weights
        self.member_weights_: Optional[np.ndarray] = None
        self.member_rmse_: Optional[np.ndarray] = None
        self.member_names_: list[str] = []

    @property
    def supports_uncertainty(self) -> bool:
        return False

    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _make_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)

        val_fraction = float(self.config.validation_fraction)
        val_fraction = min(max(val_fraction, 0.0), 0.5)

        n_val = int(round(n * val_fraction))

        # keep at least 1 train sample and 1 val sample when possible
        if n >= 3:
            n_val = max(1, min(n_val, n - 1))
        else:
            n_val = 0

        rng = np.random.default_rng(self.config.random_state)
        indices = np.arange(n)
        rng.shuffle(indices)

        if n_val == 0:
            return X, y, X, y

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if not self.members:
            raise ValueError("EnsembleSurrogate needs at least one member model.")

        X_fit, y_fit, X_val, y_val = self._make_split(X, y)

        validation_rmses = []
        self.member_names_ = []

        # Step 1: estimate member quality on internal validation split
        for prototype in self.members:
            temp_model = copy.deepcopy(prototype)
            temp_model.fit(X_fit, y_fit)
            y_pred_val = temp_model.predict(X_val)
            rmse_val = self._rmse(y_val, y_pred_val)

            validation_rmses.append(rmse_val)
            self.member_names_.append(getattr(temp_model, "name", type(temp_model).__name__))

        self.member_rmse_ = np.asarray(validation_rmses, dtype=float)

        # Step 2: convert errors to weights
        inv = 1.0 / (self.member_rmse_ + float(self.config.weight_epsilon))
        self.member_weights_ = inv / np.sum(inv)

        # Step 3: fit fresh copies on full data for final prediction
        self.trained_members_ = []
        for prototype in self.members:
            final_model = copy.deepcopy(prototype)
            final_model.fit(X, y)
            self.trained_members_.append(final_model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if not self.trained_members_:
            raise RuntimeError("EnsembleSurrogate is not fitted yet.")

        if self.member_weights_ is None:
            raise RuntimeError("EnsembleSurrogate weights are not available.")

        preds = [
            np.asarray(model.predict(X), dtype=float).ravel()
            for model in self.trained_members_
        ]
        pred_matrix = np.vstack(preds)  # shape: (n_members, n_samples)

        return np.average(pred_matrix, axis=0, weights=self.member_weights_)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        y = self.predict(X)
        return y, None

