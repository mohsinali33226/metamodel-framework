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
    out_path = Path(le["uq_predictions_gp"])

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
    out = pd.DataFrame({"sample_idx": np.arange(len(X_test_df))})

    for _, r in gp_rows.iterrows():
        output_name = str(r["output"])
        winner = str(r["winner"])

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

        out[f"y_true__{output_name}"] = y_true
        out[f"y_mean__{output_name}"] = y_mean
        out[f"y_std__{output_name}"] = y_std

        print("Exported:", output_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Shape: {out.shape}")


if __name__ == "__main__":
    main()

