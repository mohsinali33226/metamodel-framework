from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make src/ importable when running scripts from project root
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from framework.utils.paths import load_output_paths


def main() -> None:
    cfg = load_output_paths()
    le = cfg["l_effective"]

    registry_path = Path(le["registry"])
    models_dir = Path(le["models_dir"])
    x_test_path = Path(le["x_test"])
    y_test_path = Path(le["y_test"])
    out_path = Path(le["uq_coverage"])

    if not registry_path.exists():
        raise FileNotFoundError(f"Missing: {registry_path}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Missing models folder: {models_dir}")
    if not x_test_path.exists():
        raise FileNotFoundError(f"Missing: {x_test_path}")
    if not y_test_path.exists():
        raise FileNotFoundError(f"Missing: {y_test_path}")

    reg = pd.read_csv(registry_path)
    X_test_df = pd.read_csv(x_test_path)
    Y_test = pd.read_csv(y_test_path)

    gp_rows = reg[reg["winner"].astype(str).str.upper() == "GP"].copy()
    print("GP outputs found:", len(gp_rows))
    if len(gp_rows) == 0:
        raise ValueError("No GP winners found in registry.")

    X_np = X_test_df.to_numpy()
    rows = []

    for _, r in gp_rows.iterrows():
        output_name = str(r["output"])
        winner = str(r["winner"])

        print("Processing:", output_name)

        model_file = models_dir / f"{winner}__{output_name}.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model file: {model_file}")

        model = joblib.load(model_file)

        if not hasattr(model, "predict_with_uncertainty"):
            raise ValueError(f"Model for {output_name} does not support predict_with_uncertainty().")

        y_mean, y_std = model.predict_with_uncertainty(X_np)

        if y_std is None:
            raise ValueError(f"GP model for {output_name} returned y_std=None.")

        if output_name not in Y_test.columns:
            raise ValueError(f"Y_test does not contain output column '{output_name}'")

        y_true = Y_test[output_name].to_numpy(dtype=float)
        y_mean = np.asarray(y_mean, dtype=float).reshape(-1)
        y_std = np.asarray(y_std, dtype=float).reshape(-1)

        lower = y_mean - 2.0 * y_std
        upper = y_mean + 2.0 * y_std
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))

        rows.append(
            {
                "output": output_name,
                "winner": winner,
                "n_test": int(len(y_true)),
                "coverage_pm_2std": coverage,
                "std_median": float(np.median(y_std)),
                "std_mean": float(np.mean(y_std)),
                "y_true_min": float(np.min(y_true)),
                "y_true_max": float(np.max(y_true)),
            }
        )

    print("Rows collected:", len(rows))

    df = pd.DataFrame(rows).sort_values("output").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    cov = df["coverage_pm_2std"].to_numpy(dtype=float)
    print(f"Saved UQ coverage report: {out_path}")
    print("Coverage (mean Â± 2*std) summary:")
    print(f"  median={np.median(cov):.4f}, min={np.min(cov):.4f}, max={np.max(cov):.4f}")
    print(f"  n_gp_outputs={len(df)}")


if __name__ == "__main__":
    main()

