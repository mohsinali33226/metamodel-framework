import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


Xtr = pd.read_csv("./data/outputs/l_effective/X_train.csv")
Xva = pd.read_csv("./data/outputs/l_effective/X_val.csv")
Xte = pd.read_csv("./data/outputs/l_effective/X_test.csv")

Ytr = pd.read_csv("./data/outputs/l_effective/Y_train.csv")
Yva = pd.read_csv("./data/outputs/l_effective/Y_val.csv")
Yte = pd.read_csv("./data/outputs/l_effective/Y_test.csv")

Xdev = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
Ydev = pd.concat([Ytr, Yva], axis=0).reset_index(drop=True)

targets = [
    "eff_low_X",
    "eff_mid_X",
    "eff_low_RotY",
    "eff_low_RotZ",
    "eff_broad_X",
    "eff_high_X",
    "eff_mid_RotY",
    "eff_broad_Z",
]

current_best = {
    "eff_low_X": 0.25777,
    "eff_mid_X": 0.35422,
    "eff_low_RotY": 0.45813,
    "eff_low_RotZ": 0.50252,
    "eff_broad_X": 0.56163,
    "eff_high_X": 0.67435,
    "eff_mid_RotY": 0.69835,
    "eff_broad_Z": 0.78698,
}


def make_model(name):
    if name == "XGB_slow_depth3":
        return XGBRegressor(
            n_estimators=1200,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if name == "XGB_med_depth4":
        return XGBRegressor(
            n_estimators=900,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if name == "XGB_reg_depth5":
        return XGBRegressor(
            n_estimators=700,
            max_depth=5,
            learning_rate=0.035,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if name == "LGBM_slow_leaves15":
        return LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.02,
            num_leaves=15,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    if name == "LGBM_med_leaves31":
        return LGBMRegressor(
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    if name == "CAT_slow_depth4":
        return CatBoostRegressor(
            iterations=1000,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

    if name == "CAT_med_depth5":
        return CatBoostRegressor(
            iterations=800,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=6.0,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

    raise ValueError(name)


model_names = [
    "XGB_slow_depth3",
    "XGB_med_depth4",
    "XGB_reg_depth5",
    "LGBM_slow_leaves15",
    "LGBM_med_leaves31",
    "CAT_slow_depth4",
    "CAT_med_depth5",
]

rows = []

for target in targets:
    print("\n" + "=" * 70)
    print("TARGET:", target)
    print("current_best_test_R2:", current_best[target])

    ytr = Ytr[target]
    yva = Yva[target]
    ydev = Ydev[target]
    yte = Yte[target]

    best_val_r2 = -999
    best_name = None

    for name in model_names:
        model = make_model(name)
        model.fit(Xtr, ytr)
        val_pred = model.predict(Xva)
        val_r2 = r2_score(yva, val_pred)
        val_mae = mean_absolute_error(yva, val_pred)

        print(name, "VAL_R2=", round(val_r2, 5), "VAL_MAE=", round(val_mae, 3))

        rows.append({
            "target": target,
            "stage": "validation",
            "model": name,
            "val_r2": val_r2,
            "val_mae": val_mae,
            "test_r2": np.nan,
            "test_mae": np.nan,
            "current_best": current_best[target],
            "improved": np.nan,
        })

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_name = name

    print("BEST_VALIDATION_MODEL:", best_name, "VAL_R2=", round(best_val_r2, 5))

    final_model = make_model(best_name)
    final_model.fit(Xdev, ydev)
    test_pred = final_model.predict(Xte)

    test_r2 = r2_score(yte, test_pred)
    test_mae = mean_absolute_error(yte, test_pred)
    improved = test_r2 > current_best[target]

    print("FINAL_TEST_MODEL:", best_name)
    print("TEST_R2=", round(test_r2, 5), "TEST_MAE=", round(test_mae, 3), "IMPROVED=", improved)

    rows.append({
        "target": target,
        "stage": "final_test_after_val_selection",
        "model": best_name,
        "val_r2": best_val_r2,
        "val_mae": np.nan,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "current_best": current_best[target],
        "improved": improved,
    })

out = "./data/outputs/l_effective/direct_boosting_sweep_priority_targets.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("\nSaved:", out)

