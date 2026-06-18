from __future__ import annotations

import sys
from pathlib import Path

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

    dataset_path = Path(le["dataset"])
    out_x = Path(le["x_inputs"])
    out_y = Path(le["y_outputs"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # outputs are eff_*
    y_cols = [c for c in df.columns if c.startswith("eff_")]
    if len(y_cols) != 20:
        print("Warning: expected 20 eff_ outputs, found:", len(y_cols))

    # inputs = everything else except file_name
    x_cols = [c for c in df.columns if (c not in y_cols) and (c != "file_name")]

    X = df[x_cols].copy()
    Y = df[y_cols].copy()

    # drop t_s if present
    if "t_s" in X.columns:
        X = X.drop(columns=["t_s"])
        print("Dropped column: t_s")
    else:
        print("t_s not found in X (nothing dropped)")

    out_x.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_x, index=False)
    Y.to_csv(out_y, index=False)

    print("Wrote:", out_x, "shape:", X.shape)
    print("Wrote:", out_y, "shape:", Y.shape)
    print("X columns:", list(X.columns))


if __name__ == "__main__":
    main()

