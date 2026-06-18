from __future__ import annotations

import sys
import json
from pathlib import Path

import joblib
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

    x_test_path = Path(le["x_test"])
    registry_path = Path(le["registry"])
    models_dir = Path(le["models_dir"])
    pred_path = Path(le["pred_test"])
    meta_path = pred_path.parent / "pred_test_meta.json"

    if not x_test_path.exists():
        raise FileNotFoundError(f"Missing: {x_test_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing: {registry_path}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Missing models folder: {models_dir}")

    X_test_df = pd.read_csv(x_test_path)
    registry = pd.read_csv(registry_path)

    required_cols = {"output", "winner"}
    missing = required_cols - set(registry.columns)
    if missing:
        raise ValueError(f"Registry missing columns {sorted(missing)}. Found: {list(registry.columns)}")

    X_test = X_test_df.to_numpy()

    preds = pd.DataFrame(index=range(len(X_test_df)))
    models_used = []

    for _, row in registry.iterrows():
        output_name = str(row["output"])
        winner = str(row["winner"])

        model_file = models_dir / f"{winner}__{output_name}.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model file for '{output_name}': {model_file}")

        model = joblib.load(model_file)
        y_pred = model.predict(X_test)

        if hasattr(y_pred, "reshape"):
            y_pred = y_pred.reshape(-1)

        preds[output_name] = y_pred
        models_used.append({
            "output": output_name,
            "winner": winner,
            "model_file": model_file.name,
        })

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path, index=False)

    meta = {
        "x_test_path": str(x_test_path),
        "registry_path": str(registry_path),
        "models_dir": str(models_dir),
        "pred_path": str(pred_path),
        "n_samples": int(len(X_test_df)),
        "n_outputs": int(preds.shape[1]),
        "outputs": list(preds.columns),
        "models_used": models_used,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved predictions: {pred_path}")
    print(f"Saved meta:        {meta_path}")
    print(f"Pred shape:        {preds.shape} (rows, outputs)")


if __name__ == "__main__":
    main()

