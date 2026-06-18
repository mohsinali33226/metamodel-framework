from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Make src/ importable when running scripts from project root
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from framework.utils.paths import load_output_paths


RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def main() -> None:
    cfg = load_output_paths()
    le = cfg["l_effective"]

    x_path = Path(le["x_inputs"])
    y_path = Path(le["y_outputs"])
    out_dir = Path(le["dir"])

    if not x_path.exists():
        raise FileNotFoundError(f"Missing: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing: {y_path}")

    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)

    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    val_frac_rel = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=val_frac_rel, random_state=RANDOM_STATE, shuffle=True
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(Path(le["x_train"]), index=False)
    Y_train.to_csv(Path(le["y_train"]), index=False)

    X_val.to_csv(Path(le["x_val"]), index=False)
    Y_val.to_csv(Path(le["y_val"]), index=False)

    X_test.to_csv(Path(le["x_test"]), index=False)
    Y_test.to_csv(Path(le["y_test"]), index=False)

    print("Train:", X_train.shape, Y_train.shape)
    print("Val:  ", X_val.shape, Y_val.shape)
    print("Test: ", X_test.shape, Y_test.shape)


if __name__ == "__main__":
    main()

