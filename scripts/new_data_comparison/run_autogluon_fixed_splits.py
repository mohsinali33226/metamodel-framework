from pathlib import Path
import shutil
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from autogluon.tabular import TabularPredictor


SPLIT_DIR = Path("data/outputs/new_data/fixed_splits")
OUT = Path("data/outputs/new_data/autogluon_fixed_split")
PRED_OUT = OUT / "predictions"
LEADERBOARD_OUT = OUT / "leaderboards"
MODEL_OUT = OUT / "models"

OUT.mkdir(parents=True, exist_ok=True)
PRED_OUT.mkdir(parents=True, exist_ok=True)
LEADERBOARD_OUT.mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)

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

PRESETS_BY_DATASET = {
    "G28": "medium_quality",
    "G200": "best_quality",
}

TIME_LIMIT_BY_DATASET = {
    "G28": 60,
    "G200": 300,
}


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def prepare_autogluon_data(train_df, test_df, target):
    feature_cols = [
        c for c in train_df.columns
        if c not in TARGETS_YZ and c not in ["group", "id"]
    ]

    use_train = train_df[feature_cols + [target]].copy()
    use_test = test_df[feature_cols + [target]].copy()

    return use_train, use_test, feature_cols


def run_dataset(dataset):
    print("\n" + "=" * 80)
    print("DATASET:", dataset)

    train_df = pd.read_csv(SPLIT_DIR / f"{dataset}_train.csv")
    test_df = pd.read_csv(SPLIT_DIR / f"{dataset}_test.csv")

    rows = []

    for target in TARGETS_YZ:
        print("\nTARGET:", target)

        train_data, test_data, feature_cols = prepare_autogluon_data(
            train_df,
            test_df,
            target,
        )

        model_path = MODEL_OUT / f"{dataset}_{target}"

        if model_path.exists():
            shutil.rmtree(model_path)

        predictor = TabularPredictor(
            label=target,
            eval_metric="r2",
            path=str(model_path),
            verbosity=0,
        )

        predictor.fit(
            train_data,
            presets=PRESETS_BY_DATASET[dataset],
            time_limit=TIME_LIMIT_BY_DATASET[dataset],
        )

        X_test = test_data.drop(columns=[target])
        y_test = test_data[target]

        pred = predictor.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        test_rmse = rmse(y_test, pred)

        leaderboard = predictor.leaderboard(test_data, silent=True)
        leaderboard_path = LEADERBOARD_OUT / f"leaderboard_{dataset}_{target}.csv"
        leaderboard.to_csv(leaderboard_path, index=False)

        pred_df = pd.DataFrame({
            "dataset": dataset,
            "target": target,
            "y_true": y_test.values,
            "pred_autogluon": pred.values,
        })

        pred_path = PRED_OUT / f"autogluon_predictions_{dataset}_{target}.csv"
        pred_df.to_csv(pred_path, index=False)

        row = {
            "dataset": dataset,
            "target": target,
            "features": len(feature_cols),
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "preset": PRESETS_BY_DATASET[dataset],
            "time_limit": TIME_LIMIT_BY_DATASET[dataset],
            "eval_metric": "r2",
            "best_model": predictor.model_best,
            "test_r2": r2,
            "test_mae": mae,
            "test_rmse": test_rmse,
            "model_path": str(model_path),
            "leaderboard_path": str(leaderboard_path),
            "prediction_path": str(pred_path),
        }

        rows.append(row)

        print(
            f"AutoGluon R2={r2:.5f} "
            f"MAE={mae:.5f} "
            f"RMSE={test_rmse:.5f} "
            f"best={predictor.model_best}"
        )

    return rows


all_rows = []

for dataset in ["G28", "G200"]:
    all_rows.extend(run_dataset(dataset))

results = pd.DataFrame(all_rows)

out_file = OUT / "autogluon_fixed_split_results.csv"
results.to_csv(out_file, index=False)

print("\nDONE AUTOGLUON FIXED-SPLIT BENCHMARK")
print(
    results[
        [
            "dataset",
            "target",
            "preset",
            "time_limit",
            "best_model",
            "test_r2",
            "test_mae",
            "test_rmse",
        ]
    ].to_string(index=False)
)

print("\nsaved:", out_file)
print("saved predictions:", PRED_OUT)
print("saved leaderboards:", LEADERBOARD_OUT)
print("saved models:", MODEL_OUT)



