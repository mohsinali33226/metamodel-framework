from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# Make src/ importable when running scripts from project root
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from framework.utils.paths import load_output_paths


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.0) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def summarize(df: pd.DataFrame, col: str) -> str:
    v = df[col].to_numpy(dtype=float)
    return f"{col}: median={np.median(v):.6g}, min={np.min(v):.6g}, max={np.max(v):.6g}"


def main() -> None:
    cfg = load_output_paths()
    le = cfg["l_effective"]

    y_test_path = Path(le["y_test"])
    pred_path = Path(le["pred_test"])
    out_metrics_path = Path(le["pred_test_metrics"])

    if not y_test_path.exists():
        raise FileNotFoundError(f"Missing: {y_test_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing: {pred_path}")

    Y_true = pd.read_csv(y_test_path)
    Y_pred = pd.read_csv(pred_path)

    missing = [c for c in Y_true.columns if c not in Y_pred.columns]
    if missing:
        raise ValueError(f"Predictions missing targets: {missing}")

    Y_pred = Y_pred[Y_true.columns]

    rows = []
    for col in Y_true.columns:
        yt = Y_true[col].to_numpy(dtype=float)
        yp = Y_pred[col].to_numpy(dtype=float)

        mse = mean_squared_error(yt, yp)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))

        mape = mape_percent(yt, yp, eps=0.0)
        smape = smape_percent(yt, yp)

        rows.append(
            {
                "output": col,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape_percent": mape,
                "smape_percent": smape,
                "y_true_min": float(np.min(yt)),
                "y_true_max": float(np.max(yt)),
                "y_true_mean": float(np.mean(yt)),
                "y_true_std": float(np.std(yt)),
            }
        )

    df = pd.DataFrame(rows).sort_values("output").reset_index(drop=True)
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_metrics_path, index=False)

    print(f"Saved metrics: {out_metrics_path}")
    print(summarize(df, "rmse"))
    print(summarize(df, "mae"))
    print(summarize(df, "r2"))
    print(summarize(df, "mape_percent"))
    print(summarize(df, "smape_percent"))

    summary_json = out_metrics_path.parent / "pred_test_metrics_summary.json"
    summary = {
        "metrics_file": str(out_metrics_path),
        "n_outputs": int(df.shape[0]),
        "rmse_median": float(np.median(df["rmse"])),
        "mae_median": float(np.median(df["mae"])),
        "r2_median": float(np.median(df["r2"])),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary:  {summary_json}")


if __name__ == "__main__":
    main()

