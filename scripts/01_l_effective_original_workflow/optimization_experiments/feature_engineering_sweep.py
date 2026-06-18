import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error


def add_features(df):
    df = df.copy()
    eps = 1e-9

    def has(*cols):
        return all(c in df.columns for c in cols)

    if has("Fr", "Fa"):
        df["F_resultant"] = np.sqrt(df["Fr"] ** 2 + df["Fa"] ** 2)
        df["Fa_Fr_ratio"] = df["Fa"] / (df["Fr"].abs() + eps)
        df["Fr_Fa_product"] = df["Fr"] * df["Fa"]

    if has("Fr", "Z"):
        df["Fr_per_Z"] = df["Fr"] / (df["Z"].abs() + eps)

    if has("Fa", "Z"):
        df["Fa_per_Z"] = df["Fa"] / (df["Z"].abs() + eps)

    if has("Fr", "Dw"):
        df["Fr_per_Dw"] = df["Fr"] / (df["Dw"].abs() + eps)

    if has("Fa", "Dw"):
        df["Fa_per_Dw"] = df["Fa"] / (df["Dw"].abs() + eps)

    if has("n", "Fr"):
        df["n_Fr"] = df["n"] * df["Fr"]

    if has("n", "Fa"):
        df["n_Fa"] = df["n"] * df["Fa"]

    if has("n", "F_resultant"):
        df["n_F_resultant"] = df["n"] * df["F_resultant"]

    if has("n", "T"):
        df["n_T"] = df["n"] * df["T"]

    if has("n", "Dw"):
        df["n_Dw"] = df["n"] * df["Dw"]

    if has("T", "e_c_norm"):
        df["T_e_c_norm"] = df["T"] * df["e_c_norm"]

    if has("n", "e_c_norm"):
        df["n_e_c_norm"] = df["n"] * df["e_c_norm"]

    if has("dSi", "dSa"):
        df["dS_mean"] = 0.5 * (df["dSi"] + df["dSa"])
        df["dS_diff"] = df["dSa"] - df["dSi"]
        df["dS_product"] = df["dSi"] * df["dSa"]

    if has("SVi", "SVa"):
        df["SV_mean"] = 0.5 * (df["SVi"] + df["SVa"])
        df["SV_diff"] = df["SVa"] - df["SVi"]
        df["SV_product"] = df["SVi"] * df["SVa"]

    if has("t_roundness_ir", "t_roundness_or"):
        df["roundness_sum"] = df["t_roundness_ir"] + df["t_roundness_or"]
        df["roundness_diff"] = df["t_roundness_or"] - df["t_roundness_ir"]
        df["roundness_product"] = df["t_roundness_ir"] * df["t_roundness_or"]

    if has("Ra_w", "SV_mean"):
        df["Ra_w_SV_mean"] = df["Ra_w"] * df["SV_mean"]

    if has("Z", "Dw"):
        df["Z_Dw"] = df["Z"] * df["Dw"]

    for col in ["Fr", "Fa", "n", "T", "e_c_norm", "Dw", "Ra_w"]:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2

    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


Xtr = pd.read_csv("./data/outputs/l_effective/X_train.csv")
Xva = pd.read_csv("./data/outputs/l_effective/X_val.csv")
Xte = pd.read_csv("./data/outputs/l_effective/X_test.csv")

Ytr = pd.read_csv("./data/outputs/l_effective/Y_train.csv")
Yva = pd.read_csv("./data/outputs/l_effective/Y_val.csv")
Yte = pd.read_csv("./data/outputs/l_effective/Y_test.csv")

Xtr_fe = add_features(Xtr)
Xva_fe = add_features(Xva)
Xte_fe = add_features(Xte)

Xdev_fe = pd.concat([Xtr_fe, Xva_fe], axis=0).reset_index(drop=True)
Ydev = pd.concat([Ytr, Yva], axis=0).reset_index(drop=True)

current_best = {
    "eff_mid_Z": 0.78235,
    "eff_broad_Z": 0.78698,
    "eff_broad_Y": 0.75151,
    "eff_mid_Y": 0.74857,
    "eff_high_Z": 0.73180,
    "eff_high_Y": 0.72972,
    "eff_broad_RotZ": 0.69328,
    "eff_mid_RotY": 0.69835,
    "eff_high_X": 0.67435,
    "eff_mid_RotZ": 0.67095,
    "eff_broad_RotY": 0.67344,
    "eff_high_RotZ": 0.66072,
    "eff_high_RotY": 0.62304,
    "eff_broad_X": 0.56163,
    "eff_low_Y": 0.52974,
    "eff_low_RotZ": 0.50252,
    "eff_low_Z": 0.48333,
    "eff_low_RotY": 0.45813,
    "eff_mid_X": 0.35422,
    "eff_low_X": 0.25777,
}

models = {
    "ExtraTrees_FE_800_leaf2": ExtraTreesRegressor(
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features=0.8,
    ),
    "ExtraTrees_FE_1200_leaf1": ExtraTreesRegressor(
        n_estimators=1200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features=0.7,
    ),
    "RandomForest_FE_800_leaf2": RandomForestRegressor(
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features=0.8,
    ),
    "HistGBR_FE_lr003": HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=42,
    ),
}

rows = []

print("Original feature count:", Xtr.shape[1])
print("Engineered feature count:", Xtr_fe.shape[1])

for target, old_best in current_best.items():
    print("\n" + "=" * 70)
    print("TARGET:", target)
    print("current_best:", old_best)

    ytr = Ytr[target]
    yva = Yva[target]
    ydev = Ydev[target]
    yte = Yte[target]

    best_val = -999
    best_name = None

    for name, model in models.items():
        model.fit(Xtr_fe, ytr)
        val_pred = model.predict(Xva_fe)
        val_r2 = r2_score(yva, val_pred)
        val_mae = mean_absolute_error(yva, val_pred)

        print(name, "VAL_R2=", round(val_r2, 5), "VAL_MAE=", round(val_mae, 3))

        if val_r2 > best_val:
            best_val = val_r2
            best_name = name

        rows.append({
            "target": target,
            "stage": "validation",
            "model": name,
            "val_r2": val_r2,
            "test_r2": np.nan,
            "test_mae": np.nan,
            "current_best": old_best,
            "improved": np.nan,
        })

    final_model = models[best_name]
    final_model.fit(Xdev_fe, ydev)
    test_pred = final_model.predict(Xte_fe)

    test_r2 = r2_score(yte, test_pred)
    test_mae = mean_absolute_error(yte, test_pred)
    improved = test_r2 > old_best

    print("BEST_VAL_MODEL:", best_name, "VAL_R2=", round(best_val, 5))
    print("FINAL_TEST_R2=", round(test_r2, 5), "TEST_MAE=", round(test_mae, 3), "IMPROVED=", improved)

    rows.append({
        "target": target,
        "stage": "final_test_after_val_selection",
        "model": best_name,
        "val_r2": best_val,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "current_best": old_best,
        "improved": improved,
    })

out = "./data/outputs/l_effective/feature_engineering_sweep.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("\nSaved:", out)

