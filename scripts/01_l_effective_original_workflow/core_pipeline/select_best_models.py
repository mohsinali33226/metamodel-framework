from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Make src/ importable when running scripts from project root
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from framework.surrogates import (
    SVRSurrogate,
    SVRConfig,
    GPSurrogate,
    GPConfig,
    RBFSurrogate,
    RBFConfig,
    ANNTFSurrogate,
    ANNConfig,
    EnsembleSurrogate,
    EnsembleConfig,
)
from framework.utils.paths import load_output_paths


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_model_candidates():
    gp = GPSurrogate(GPConfig())
    rbf = RBFSurrogate(RBFConfig())
    svr_10 = SVRSurrogate(SVRConfig(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))

    return [
        GPSurrogate(GPConfig()),
        RBFSurrogate(RBFConfig()),
        SVRSurrogate(SVRConfig(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
        SVRSurrogate(SVRConfig(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
        SVRSurrogate(SVRConfig(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale")),
        SVRSurrogate(SVRConfig(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale")),
        SVRSurrogate(SVRConfig(kernel="rbf", C=10.0, epsilon=0.1, gamma="auto")),
        SVRSurrogate(SVRConfig(kernel="rbf", C=10.0, epsilon=0.1, gamma=0.1)),
        ANNTFSurrogate(
            ANNConfig(
                hidden_units_1=64,
                hidden_units_2=32,
                learning_rate=1e-3,
                epochs=100,
                batch_size=32,
                validation_split=0.1,
                verbose=0,
                random_state=42,
            )
        ),
        EnsembleSurrogate(
            EnsembleConfig(
                members=[gp, rbf, svr_10]
            )
        ),
    ]


def main() -> None:
    cfg = load_output_paths()
    le = cfg["l_effective"]

    x_train_path = Path(le["x_train"])
    y_train_path = Path(le["y_train"])
    x_val_path = Path(le["x_val"])
    y_val_path = Path(le["y_val"])
    out_csv = Path(le["registry"])

    for p in [x_train_path, y_train_path, x_val_path, y_val_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    X_train = pd.read_csv(x_train_path)
    Y_train = pd.read_csv(y_train_path)
    X_val = pd.read_csv(x_val_path)
    Y_val = pd.read_csv(y_val_path)

    Xtr = X_train.values
    Xva = X_val.values

    targets = list(Y_train.columns)
    print("Targets:", len(targets))

    rows = []
    t_all = time.time()

    for i, target in enumerate(targets, start=1):
        ytr = Y_train[target].values
        yva = Y_val[target].values

        print(f"\n[{i}/{len(targets)}] Target: {target}")

        best_model_info = None
        best_rmse = None

        for model in build_model_candidates():
            t0 = time.time()
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            score = rmse(yva, pred)
            elapsed = time.time() - t0

            model_label = model.name.upper()
            print(f"  {model_label}: val RMSE={score:.6g}, time={elapsed:.2f}s")

            if (best_rmse is None) or (score < best_rmse):
                best_rmse = score
                best_model_info = {
                    "output": target,
                    "winner": model_label,
                    "val_rmse": score,
                    "fit_time_sec": elapsed,
                    "model_class": type(model).__name__,
                    "model_params": repr(getattr(model, "config", "")),
                }

        rows.append(best_model_info)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\nWrote:", out_csv)
    print("Total time (min):", (time.time() - t_all) / 60.0)
    print("\nWinner counts:")
    print(df["winner"].value_counts())


if __name__ == "__main__":
    main()

