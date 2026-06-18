from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class MOPConfig:
    # metric used for ranking models (lower is better for rmse/mae, higher is better for r2)
    ranking_metric: str = "rmse"
    # weight of complexity penalty (0 disables penalty)
    complexity_lambda: float = 0.0
    # if True, prefer models with uncertainty if ranking ties happen
    prefer_uq_on_tie: bool = True


def _metric_direction(metric: str) -> str:
    m = metric.lower().strip()
    if m in ["rmse", "mae", "mape"]:
        return "min"
    if m in ["r2", "r_squared", "r^2"]:
        return "max"
    # default: minimize
    return "min"


def _complexity_penalty(model_info: Dict[str, Any]) -> float:
    """
    Very lightweight proxy for complexity (kept simple on purpose).
    Expected optional fields in model_info:
      - "n_params" (int) or
      - "train_time_sec" (float) or
      - "model_family" (str)
    """
    if "n_params" in model_info and model_info["n_params"] is not None:
        return float(model_info["n_params"])
    if "train_time_sec" in model_info and model_info["train_time_sec"] is not None:
        return float(model_info["train_time_sec"])
    # fallback family-based rough ordering
    fam = str(model_info.get("model_family", "")).lower()
    if fam in ["tf_ann", "nn", "mlp"]:
        return 3.0
    if fam in ["gpr", "gp", "kriging"]:
        return 2.0
    if fam in ["svr", "svm"]:
        return 2.0
    if fam in ["rbf"]:
        return 1.0
    return 1.5


def select_best_model_mop(
    results_by_model: Dict[str, Dict[str, float]],
    model_infos: Optional[Dict[str, Dict[str, Any]]] = None,
    cfg: Optional[MOPConfig] = None,
) -> str:
    """
    MOP-inspired selection:
      score = metric_value (+/-) + lambda * complexity_penalty

    - For minimization metrics (rmse/mae): lower score is better
    - For maximization metrics (r2): we convert to minimization via score = -r2 + penalty

    Tie-break:
      1) prefer UQ-capable (if cfg.prefer_uq_on_tie and info["has_uq"] True)
      2) then choose simpler penalty
    """
    if cfg is None:
        cfg = MOPConfig()
    if model_infos is None:
        model_infos = {}

    metric = cfg.ranking_metric.lower().strip()
    direction = _metric_direction(metric)

    best_name = None
    best_score = None

    # compute scores
    scores = {}
    for name, metrics in results_by_model.items():
        if metric not in metrics:
            raise KeyError(f"Ranking metric '{metric}' not found for model '{name}'. Available: {list(metrics.keys())}")

        val = float(metrics[metric])
        info = model_infos.get(name, {})
        penalty = cfg.complexity_lambda * _complexity_penalty(info)

        if direction == "min":
            score = val + penalty
        else:
            score = (-val) + penalty  # maximize metric by minimizing negative

        scores[name] = (score, val, penalty)

        if best_score is None or score < best_score:
            best_score = score
            best_name = name

    # tie handling (very small tolerance)
    tol = 1e-12
    tied = [n for n, (s, _, _) in scores.items() if abs(s - best_score) <= tol]

    if len(tied) <= 1:
        return best_name  # type: ignore

    # 1) prefer uncertainty-capable model if requested
    if cfg.prefer_uq_on_tie:
        uq = [n for n in tied if bool(model_infos.get(n, {}).get("has_uq", False))]
        if len(uq) == 1:
            return uq[0]
        if len(uq) > 1:
            tied = uq

    # 2) pick smallest penalty proxy
    best2 = None
    best_pen = None
    for n in tied:
        pen = float(scores[n][2])
        if best_pen is None or pen < best_pen:
            best_pen = pen
            best2 = n

    return best2  # type: ignore


