from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SamplingSpec:
    method: str = "lhs"
    n_initial: int = 30
    seed: int = 42


def lhs_sample(bounds: list[list[float]], n: int, seed: int = 42) -> np.ndarray:
    """
    Simple Latin Hypercube Sampling (LHS).
    bounds: [[low1, high1], [low2, high2], ...]
    returns: (n, dim) array
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)

    # Create n intervals in [0,1] for each dimension and sample once per interval
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, dim))
    a = cut[:n]
    b = cut[1:n + 1]
    points01 = a[:, None] + (b - a)[:, None] * u  # (n, dim)

    # Shuffle each dimension independently
    for j in range(dim):
        rng.shuffle(points01[:, j])

    # Scale to bounds
    X = np.zeros_like(points01)
    for j, (low, high) in enumerate(bounds):
        X[:, j] = low + points01[:, j] * (high - low)

    return X


