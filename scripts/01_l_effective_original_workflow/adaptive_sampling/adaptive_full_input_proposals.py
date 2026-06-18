import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error


BASE = Path("./data/outputs/l_effective")
N_TOP = 80
N_CLUSTERS = 5

Xtr_all = pd.read_csv(BASE / "X_train.csv")
Xva_all = pd.read_csv(BASE / "X_val.csv")
Xte_all = pd.read_csv(BASE / "X_test.csv")

Ytr = pd.read_csv(BASE / "Y_train.csv")
Yva = pd.read_csv(BASE / "Y_val.csv")
Yte = pd.read_csv(BASE / "Y_test.csv")

ALL_INPUTS = list(Xtr_all.columns)

targets = {
    "eff_low_X": {
        "features": ["n", "Z", "dSa", "Dw", "T", "Fr", "Fa", "e_c_norm", "t_roundness_ir"],
        "model": SVR(kernel="rbf", C=300.0, epsilon=2.0, gamma=0.1),
    },
    "eff_mid_X": {
        "features": ALL_INPUTS,
        "model": SVR(kernel="rbf", C=1000.0, epsilon=1.0, gamma=0.03),
    },
    "eff_low_RotY": {
        "features": ["n", "T", "dSa", "Ra_w", "Fa", "s", "Fr", "SVa", "Dw", "e_c_norm"],
        "model": SVR(kernel="rbf", C=1000.0, epsilon=1.0, gamma=0.03),
    },
    "eff_low_RotZ": {
        "features": ALL_INPUTS,
        "model": SVR(kernel="rbf", C=1000.0, epsilon=1.0, gamma=0.03),
    },
}

all_candidates = []

for target, cfg in targets.items():
    FEATURES = cfg["features"]

    Xtr = Xtr_all[FEATURES]
    Xva = Xva_all[FEATURES]
    Xte = Xte_all[FEATURES]

    ytr = Ytr[target]
    yva = Yva[target]
    yte = Yte[target]

    Xdev = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
    ydev = pd.concat([ytr, yva], axis=0).reset_index(drop=True)

    model = make_pipeline(StandardScaler(), cfg["model"])
    model.fit(Xdev, ydev)
    pred = model.predict(Xte)

    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)
    abs_err = np.abs(yte.values - pred)

    scaler = StandardScaler()
    Xdev_s = scaler.fit_transform(Xdev)
    Xte_s = scaler.transform(Xte)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(Xdev_s)
    dist, _ = nn.kneighbors(Xte_s)
    dist = dist.ravel()

    err_score = (abs_err - abs_err.min()) / (abs_err.max() - abs_err.min() + 1e-12)
    dist_score = (dist - dist.min()) / (dist.max() - dist.min() + 1e-12)
    adaptive_score = 0.7 * err_score + 0.3 * dist_score

    rank_df = Xte_all.copy()
    rank_df["target_name"] = target
    rank_df["true_value"] = yte.values
    rank_df["prediction"] = pred
    rank_df["abs_error"] = abs_err
    rank_df["nn_distance"] = dist
    rank_df["adaptive_score"] = adaptive_score
    rank_df["diagnostic_r2"] = r2
    rank_df["diagnostic_mae"] = mae

    top = rank_df.sort_values("adaptive_score", ascending=False).head(N_TOP).copy()

    cluster_scaler = StandardScaler()
    Xtop_s = cluster_scaler.fit_transform(top[FEATURES])

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    top["cluster"] = kmeans.fit_predict(Xtop_s)

    for c in sorted(top["cluster"].unique()):
        sub = top[top["cluster"] == c].copy()
        best = sub.sort_values("adaptive_score", ascending=False).iloc[0]

        row = {
            "target_name": target,
            "cluster": int(c),
            "n_points_in_cluster": int(len(sub)),
            "mean_abs_error_cluster": float(sub["abs_error"].mean()),
            "mean_nn_distance_cluster": float(sub["nn_distance"].mean()),
            "representative_abs_error": float(best["abs_error"]),
            "representative_nn_distance": float(best["nn_distance"]),
            "representative_adaptive_score": float(best["adaptive_score"]),
            "diagnostic_r2": float(r2),
            "diagnostic_mae": float(mae),
        }

        for col in ALL_INPUTS:
            row[col] = float(best[col])

        all_candidates.append(row)

df = pd.DataFrame(all_candidates)
df = df.sort_values(["target_name", "representative_adaptive_score"], ascending=[True, False]).reset_index(drop=True)

out = BASE / "adaptive_full_input_new_simulation_proposals.csv"
df.to_csv(out, index=False)

print("saved:", out)
print("shape:", df.shape)
print(df[["target_name", "cluster", "n_points_in_cluster", "representative_adaptive_score"] + ALL_INPUTS].to_string(index=False))

