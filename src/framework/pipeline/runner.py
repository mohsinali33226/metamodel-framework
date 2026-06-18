from __future__ import annotations

from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from framework.validation import split_indices, rmse, mae, r2
from framework.validation import MOPConfig, select_best_model_mop
from framework.surrogates.rbf import RBFSurrogate
from framework.surrogates.svr import SVRSurrogate


def _normalize_group_col(group_col):
    """
    Normalize group_col values coming from YAML:
    - None / "" / "none" / "null" / "nan" -> None
    - float nan -> None
    - otherwise -> stripped string
    """
    if group_col is None:
        return None

    if isinstance(group_col, float) and np.isnan(group_col):
        return None

    if isinstance(group_col, str):
        s = group_col.strip()
        if s.lower() in ("", "none", "null", "nan"):
            return None
        return s

    # any other type -> safest fallback
    return None


def run_validation_on_csv(
    run_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
    split_method: str = "train_test_split",
    group_col=None,
    n_splits: int = 5,
):

    # ------------------------------------------------------------
    # 1) Load saved data (keep DataFrame so we can access columns)
    # ------------------------------------------------------------
    x_path = run_dir / "X_initial.csv"
    y_path = run_dir / "y_initial.csv"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    X = x_df.values
    y = y_df.values

    # ------------------------------------------------------------
    # 2) Decide split method: GroupKFold if group_col exists
    # ------------------------------------------------------------
    group_col = _normalize_group_col(group_col)

    groups = None
    if group_col is not None:
        if group_col not in x_df.columns:
            print(f"[warn] group_col='{group_col}' not found in X_initial.csv. Falling back to train_test_split.")
            group_col = None
        else:
            groups = x_df[group_col].values

    method = split_method if split_method else ("group_kfold" if groups is not None else "train_test_split")


    train_idx, test_idx = next(
        split_indices(
            method=method,
            X=X,
            y=y,
            test_size=test_size,
            seed=seed,
            shuffle=True,
            groups=groups,
            n_splits=n_splits,
        )
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ------------------------------------------------------------
    # 3) Baseline model (mean predictor)
    # ------------------------------------------------------------
    y_pred_baseline = np.mean(y_train, axis=0, keepdims=True)
    y_pred_baseline = np.repeat(y_pred_baseline, repeats=len(y_test), axis=0)

    results = {
        "split_method": method,
        "group_col": group_col,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "baseline_rmse": rmse(y_test, y_pred_baseline),
        "baseline_mae": mae(y_test, y_pred_baseline),
        "baseline_r2": r2(y_test, y_pred_baseline),
    }

    # ------------------------------------------------------------
    # 4) GP surrogate (with UQ)
    # ------------------------------------------------------------
    try:
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(X_train.shape[1]))
            + WhiteKernel(noise_level=1e-10, noise_level_bounds="fixed")
        )

        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)

        gp_model = Pipeline([
            ("scaler", StandardScaler()),
            ("gp", gp),
        ])

        gp_model.fit(X_train, np.asarray(y_train).ravel())

        joblib.dump(gp_model, run_dir / "gp_model.pkl")

        X_test_scaled = gp_model.named_steps["scaler"].transform(X_test)
        y_pred_gp, y_std_gp = gp_model.named_steps["gp"].predict(X_test_scaled, return_std=True)

        results["gp_rmse"] = rmse(y_test, y_pred_gp)
        results["gp_mae"] = mae(y_test, y_pred_gp)
        results["gp_r2"] = r2(y_test, y_pred_gp)

        pred_df = pd.DataFrame({
            "y_true": np.asarray(y_test).ravel(),
            "y_pred_gp": np.asarray(y_pred_gp).ravel(),
            "y_std_gp": np.asarray(y_std_gp).ravel(),
        })

        pred_path = run_dir / "gp_test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved GP test predictions + uncertainty: {pred_path.resolve()}")

    except Exception as e:
        print(f"[warn] GP failed: {e}")
        results["gp_rmse"] = None
        results["gp_mae"] = None
        results["gp_r2"] = None

    # ------------------------------------------------------------
    # 5) RBF surrogate (no UQ)
    # ------------------------------------------------------------
    try:
        rbf = RBFSurrogate(kernel="thin_plate_spline")
        rbf.fit(X_train, y_train)
        joblib.dump(rbf, run_dir / "rbf_model.pkl")

        y_pred_rbf = rbf.predict(X_test)

        results["rbf_rmse"] = rmse(y_test, y_pred_rbf)
        results["rbf_mae"] = mae(y_test, y_pred_rbf)
        results["rbf_r2"] = r2(y_test, y_pred_rbf)

        rbf_df = pd.DataFrame({
            "y_true": np.asarray(y_test).ravel(),
            "y_pred_rbf": np.asarray(y_pred_rbf).ravel(),
        })

        rbf_path = run_dir / "rbf_test_predictions.csv"
        rbf_df.to_csv(rbf_path, index=False)
        print(f"Saved RBF test predictions (no uncertainty): {rbf_path.resolve()}")

    except Exception as e:
        print(f"[warn] RBF failed: {e}")
        results["rbf_rmse"] = None
        results["rbf_mae"] = None
        results["rbf_r2"] = None

    # ------------------------------------------------------------
    # 6) SVR surrogate (no UQ)
    # ------------------------------------------------------------
    try:
        svr = SVRSurrogate()
        svr.fit(X_train, y_train)
        joblib.dump(svr, run_dir / "svr_model.pkl")

        y_pred_svr = svr.predict(X_test)

        results["svr_rmse"] = rmse(y_test, y_pred_svr)
        results["svr_mae"] = mae(y_test, y_pred_svr)
        results["svr_r2"] = r2(y_test, y_pred_svr)

        svr_df = pd.DataFrame({
            "y_true": np.asarray(y_test).ravel(),
            "y_pred_svr": np.asarray(y_pred_svr).ravel(),
        })

        svr_path = run_dir / "svr_test_predictions.csv"
        svr_df.to_csv(svr_path, index=False)
        print(f"Saved SVR test predictions (no uncertainty): {svr_path.resolve()}")

    except Exception as e:
        print(f"[warn] SVR failed: {e}")
        results["svr_rmse"] = None
        results["svr_mae"] = None
        results["svr_r2"] = None

    # ------------------------------------------------------------
    # 7) Save summary
    # ------------------------------------------------------------
    summary_path = run_dir / "metrics_summary.csv"
    pd.DataFrame([results]).to_csv(summary_path, index=False)
    print(f"Saved metrics summary: {summary_path.resolve()}")

    # ------------------------------------------------------------
    # 8) MOP-style automatic model selection
    # ------------------------------------------------------------
    mop_cfg = MOPConfig(
        ranking_metric="rmse",
        complexity_lambda=0.0,
        prefer_uq_on_tie=True,
    )

    results_by_model = {
        "gp":  {"rmse": results.get("gp_rmse"),  "mae": results.get("gp_mae"),  "r2": results.get("gp_r2")},
        "rbf": {"rmse": results.get("rbf_rmse"), "mae": results.get("rbf_mae"), "r2": results.get("rbf_r2")},
        "svr": {"rmse": results.get("svr_rmse"), "mae": results.get("svr_mae"), "r2": results.get("svr_r2")},
        "baseline": {"rmse": results.get("baseline_rmse"), "mae": results.get("baseline_mae"), "r2": results.get("baseline_r2")},
    }

    model_infos = {
        "gp":  {"model_family": "gpr", "has_uq": True},
        "rbf": {"model_family": "rbf", "has_uq": False},
        "svr": {"model_family": "svr", "has_uq": False},
        "baseline": {"model_family": "baseline", "has_uq": False},
    }

    best_model = select_best_model_mop(
        results_by_model=results_by_model,
        model_infos=model_infos,
        cfg=mop_cfg,
    )

    best_score = results_by_model[best_model][mop_cfg.ranking_metric]
    best_path = run_dir / "best_model.txt"
    best_path.write_text(f"{best_model}\n{mop_cfg.ranking_metric}={best_score}\n", encoding="utf-8")
    report_path = run_dir / "model_selection_report.txt"

    lines = []
    lines.append("MOP-style model selection report")
    lines.append("--------------------------------")
    lines.append(f"split_method: {method}")
    lines.append(f"n_train: {len(X_train)} | n_test: {len(X_test)}")
    lines.append("")
    lines.append("Candidate models and RMSE:")
    for m in ["gp", "rbf", "svr"]:
        rmse_val = results_by_model.get(m, {}).get("rmse", None)
        lines.append(f"  - {m}: rmse={rmse_val}")

    lines.append("")
    lines.append(f"Ranking metric: {mop_cfg.ranking_metric}")
    lines.append(f"prefer_uq_on_tie: {mop_cfg.prefer_uq_on_tie}")
    lines.append("")
    lines.append(f"Selected best model: {best_model}")
    lines.append(f"Best {mop_cfg.ranking_metric}: {best_score}")
    lines.append("")
    lines.append("Saved artifacts:")
    lines.append(f"  - metrics_summary.csv")
    lines.append(f"  - best_model.txt")
    lines.append(f"  - gp_model.pkl / rbf_model.pkl / svr_model.pkl")
    lines.append(f"  - gp_test_predictions.csv / rbf_test_predictions.csv / svr_test_predictions.csv")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved model selection report: {report_path.resolve()}")


    print(f"Best model (MOP-style): {best_model} ({mop_cfg.ranking_metric}={best_score})")
    print(f"Saved best model file: {best_path.resolve()}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation + surrogate comparison on a saved run directory.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing X_initial.csv and y_initial.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_col", default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results = run_validation_on_csv(
        run_dir=run_dir,
        test_size=args.test_size,
        seed=args.seed,
        group_col=args.group_col,
        n_splits=args.n_splits,
    )

    print("Metrics summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")


