from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np


class SurrogateModel(ABC):
    """
    Common interface for all surrogate models in the framework.

    Goal:
    - Same fit/predict API for GP, RBF, ANN, ensembles, etc.
    - Optional uncertainty output when a model supports it.
    """

    name: str = "surrogate"

    def __init__(self, **params: Any):
        self.params: Dict[str, Any] = params

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SurrogateModel":
        """Train the surrogate."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean output."""
        raise NotImplementedError

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns:
        - y_mean (n,)
        - y_std  (n,) or None if not supported
        """
        return self.predict(X), None

    @property
    def supports_uncertainty(self) -> bool:
        return False


