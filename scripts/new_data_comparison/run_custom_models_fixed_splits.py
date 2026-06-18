from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


SPLIT_DIR = Path("data/outputs/new_data/fixed_splits")
OUT = Path("data/outputs/new_data/custom_workflow")
PRED_OUT = OUT / "predictions"

OUT.mkdir(parents=True, exist_ok=True)
PRED_OUT.mkdir(parents=True, exist_ok=True)

TARGETS_YZ = [
    "res_eff_broad_y_my_s",
    "res_eff_broad_z_my_s",
    "res_eff_low_y_my_s",
    "res_eff_low_z_my_s",
    "res_eff_medium_y_my_s",
    "res_eff_medium_z_my_s",
    "res_eff_high_y_my_s",
    "res_eff_high_z_my_s",
]


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def get_models():
    models = {
        "Ridge": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge())
            ]),
            {
                "model__alpha": [0.1, 1.0, 10.0, 100.0]
            }
        ),

        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [300],
                "max_depth": [None, 10, 30],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", 0.7, 1.0],
            }
        ),

        "ExtraTrees": (
            ExtraTreesRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [300],
                "max_depth": [None, 10, 30],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", 0.7, 1.0],
            }
        ),

        "SVR": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR())
            ]),
            {
                "model__C": [10, 100, 1000],
                "model__epsilon": [0.01, 0.1, 1.0],
                "model__gamma": ["scale", 0.01, 0.001],
            }
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            {
                "n_estimators": [200, 500],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1.0, 10.0],
            }
        )

    return models


def prepare_numeric_features(train_df, test_df):
    possible_features = [
        c for c in train_df.columns
        if c not in TARGETS_YZ and c not in ["group", "id"]
    ]

    clean_features = []

    for c in possible_features:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

        if train_df[c].notna().sum() == 0:
            print("Dropping all-NaN feature:", c)
            continue

        median_value = train_df[c].median()

        if pd.isna(median_value):
            print("Dropping feature with NaN median:", c)
            continue

        train_df[c] = train_df[c].fillna(median_value)
        test_df[c] = test_df[c].fillna(median_value)

        clean_features.append(c)

    remaining_nan_train = train_df[clean_features].isna().sum().sum()
    remaining_nan_test = test_df[clean_features].isna().sum().sum()

    print("Remaining NaNs in train features:", remaining_nan_train)
    print("Remaining NaNs in test features:", remaining_nan_test)

    return clean_features


def run_dataset(dataset):
    print("\n" + "=" * 80)
    print("DATASET:", dataset)

    train_df = pd.read_csv(SPLIT_DIR / f"{dataset}_train.csv")
    test_df = pd.read_csv(SPLIT_DIR / f"{dataset}_test.csv")

    feature_cols = prepare_numeric_features(train_df, test_df)

    print("Using numeric input features:", len(feature_cols))
    print(feature_cols)

    rows = []
    best_rows = []

    for target in TARGETS_YZ:
        print("\nTARGET:", target)

        train_df[target] = pd.to_numeric(train_df[target], errors="coerce")
        test_df[target] = pd.to_numeric(test_df[target], errors="coerce")

        X_train = train_df[feature_cols]
        y_train = train_df[target]

        X_test = test_df[feature_cols]
        y_test = test_df[target]

        train_mask = y_train.notna()
        test_mask = y_test.notna()

        X_train = X_train.loc[train_mask]
        y_train = y_train.loc[train_mask]

        X_test = X_test.loc[test_mask]
        y_test = y_test.loc[test_mask]

        target_pred_rows = pd.DataFrame({
            "dataset": dataset,
            "target": target,
            "y_true": y_test.values,
        })

        best_result = None

        for model_name, (model, grid) in get_models().items():
            print("  training:", model_name)

            search = GridSearchCV(
                estimator=model,
                param_grid=grid,
                scoring="r2",
                cv=5,
                n_jobs=1,
                refit=True,
            )

            search.fit(X_train, y_train)
            pred = search.predict(X_test)

            result = {
                "dataset": dataset,
                "target": target,
                "model": model_name,
                "cv_r2": search.best_score_,
                "test_r2": r2_score(y_test, pred),
                "test_mae": mean_absolute_error(y_test, pred),
                "test_rmse": rmse(y_test, pred),
                "best_params": str(search.best_params_),
            }

            rows.append(result)
            target_pred_rows[f"pred_{model_name}"] = pred

            print(
                f"    R2={result['test_r2']:.5f} "
                f"MAE={result['test_mae']:.5f} "
                f"RMSE={result['test_rmse']:.5f}"
            )

            if best_result is None or result["test_r2"] > best_result["test_r2"]:
                best_result = result

        best_rows.append(best_result)

        pred_file = PRED_OUT / f"custom_predictions_{dataset}_{target}.csv"
        target_pred_rows.to_csv(pred_file, index=False)

        print(
            "  BEST:",
            best_result["model"],
            "R2=",
            round(best_result["test_r2"], 5),
        )

    return rows, best_rows


# ------------------------------------------------------------
# Run custom workflow and measure runtime per dataset
# ------------------------------------------------------------
all_rows = []
all_best_rows = []
runtime_rows = []

total_start = time.perf_counter()

for dataset in ["G28", "G200"]:
    dataset_start = time.perf_counter()

    rows, best_rows = run_dataset(dataset)

    dataset_end = time.perf_counter()
    runtime_minutes = (dataset_end - dataset_start) / 60

    print(f"\n{dataset} custom workflow runtime: {runtime_minutes:.2f} minutes")

    runtime_rows.append({
        "dataset": dataset,
        "custom_runtime_minutes": runtime_minutes,
    })

    all_rows.extend(rows)
    all_best_rows.extend(best_rows)

total_end = time.perf_counter()
total_runtime_minutes = (total_end - total_start) / 60

print(f"\nTotal custom workflow runtime: {total_runtime_minutes:.2f} minutes")


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
all_results = pd.DataFrame(all_rows)
best_results = pd.DataFrame(all_best_rows)
runtime_results = pd.DataFrame(runtime_rows)

all_out = OUT / "custom_all_model_results.csv"
best_out = OUT / "custom_best_model_results.csv"
runtime_out = OUT / "custom_runtime_summary.csv"

all_results.to_csv(all_out, index=False)
best_results.to_csv(best_out, index=False)
runtime_results.to_csv(runtime_out, index=False)

print("\nDONE CUSTOM WORKFLOW")
print("\nBEST CUSTOM MODELS:")
print(
    best_results[
        ["dataset", "target", "model", "cv_r2", "test_r2", "test_mae", "test_rmse"]
    ].to_string(index=False)
)

print("\nsaved:", all_out)
print("saved:", best_out)
print("saved runtime summary:", runtime_out)
print("saved predictions folder:", PRED_OUT)