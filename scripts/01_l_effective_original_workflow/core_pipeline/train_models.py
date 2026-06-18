from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd

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


def build_model_from_registry_row(row):
    winner = str(row.winner).upper()
    params_text = str(getattr(row, "model_params", "")).strip()

    if winner == "GP":
        return GPSurrogate(GPConfig())

    if winner == "RBF":
        return RBFSurrogate(RBFConfig())

    if winner == "SVR":
        if "C=1.0" in params_text and "epsilon=0.1" in params_text:
            return SVRSurrogate(SVRConfig(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"))

        if "C=100.0" in params_text and "epsilon=0.1" in params_text:
            return SVRSurrogate(SVRConfig(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale"))

        if "C=10.0" in params_text and "epsilon=0.01" in params_text:
            return SVRSurrogate(SVRConfig(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))

        return SVRSurrogate(SVRConfig(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))

    if winner == "ANN_TF":
        return ANNTFSurrogate(
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
        )

    if winner == "ENSEMBLE":
        gp = GPSurrogate(GPConfig())
        rbf = RBFSurrogate(RBFConfig())
        svr_10 = SVRSurrogate(SVRConfig(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))

        return EnsembleSurrogate(
            EnsembleConfig(
                members=[gp, rbf, svr_10],
                validation_fraction=0.2,
                random_state=42,
                weight_epsilon=1e-12,
            )
        )

    raise ValueError(f"Unknown winner: {winner}")


def main() -> None:
    cfg = load_output_paths()
    le = cfg["l_effective"]

    x_train_path = Path(le["x_train"])
    y_train_path = Path(le["y_train"])
    registry_path = Path(le["registry"])
    out_dir = Path(le["models_dir"])

    for p in [x_train_path, y_train_path, registry_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    X_train = pd.read_csv(x_train_path)
    Y_train = pd.read_csv(y_train_path)
    reg = pd.read_csv(registry_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr = X_train.values
    targets = reg["output"].tolist()

    meta_lines = []
    t_all = time.time()

    for i, row in enumerate(reg.itertuples(index=False), start=1):
        target = row.output
        winner = row.winner

        ytr = Y_train[target].values
        print(f"[{i}/{len(targets)}] Train {winner} for: {target}")

        t0 = time.time()
        model = build_model_from_registry_row(row)
        model.fit(Xtr, ytr)
        t_train = time.time() - t0

        out_path = out_dir / f"{winner}__{target}.joblib"
        joblib.dump(model, out_path)

        meta_lines.append(
            f"{target}\t{winner}\ttrain_time_s={t_train:.3f}\tfile={out_path}"
        )
        print(f"   saved: {out_path} (time={t_train:.2f}s)")

    meta_path = out_dir / "training_meta.txt"
    meta_path.write_text("\n".join(meta_lines), encoding="utf-8")

    print("\nDone.")
    print("Models folder:", out_dir)
    print("Meta:", meta_path)
    print("Total time (min):", (time.time() - t_all) / 60.0)


if __name__ == "__main__":
    main()

