from pathlib import Path
import pandas as pd

BASE = Path("data/outputs/new_data")

CUSTOM_FILE = BASE / "custom_workflow" / "custom_best_model_results.csv"
AUTOGLUON_FILE = BASE / "autogluon_fixed_split" / "autogluon_fixed_split_results.csv"

OUT = BASE / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

custom = pd.read_csv(CUSTOM_FILE)
ag = pd.read_csv(AUTOGLUON_FILE)

custom_small = custom.rename(columns={
    "model": "custom_best_model",
    "test_r2": "custom_r2",
    "test_mae": "custom_mae",
    "test_rmse": "custom_rmse",
    "cv_r2": "custom_cv_r2",
})[
    [
        "dataset",
        "target",
        "custom_best_model",
        "custom_cv_r2",
        "custom_r2",
        "custom_mae",
        "custom_rmse",
    ]
]

ag_small = ag.rename(columns={
    "best_model": "autogluon_best_model",
    "test_r2": "autogluon_r2",
    "test_mae": "autogluon_mae",
    "test_rmse": "autogluon_rmse",
})[
    [
        "dataset",
        "target",
        "preset",
        "time_limit",
        "autogluon_best_model",
        "autogluon_r2",
        "autogluon_mae",
        "autogluon_rmse",
    ]
]

comparison = custom_small.merge(
    ag_small,
    on=["dataset", "target"],
    how="inner",
)

comparison["r2_difference_autogluon_minus_custom"] = (
    comparison["autogluon_r2"] - comparison["custom_r2"]
)

comparison["mae_difference_autogluon_minus_custom"] = (
    comparison["autogluon_mae"] - comparison["custom_mae"]
)

comparison["rmse_difference_autogluon_minus_custom"] = (
    comparison["autogluon_rmse"] - comparison["custom_rmse"]
)

comparison["winner_r2"] = comparison.apply(
    lambda row: "AutoGluon" if row["autogluon_r2"] > row["custom_r2"] else "Custom",
    axis=1,
)

comparison = comparison.sort_values(["dataset", "target"]).reset_index(drop=True)

summary = (
    comparison.groupby("dataset")
    .agg(
        custom_mean_r2=("custom_r2", "mean"),
        autogluon_mean_r2=("autogluon_r2", "mean"),
        mean_r2_gain=("r2_difference_autogluon_minus_custom", "mean"),
        custom_median_r2=("custom_r2", "median"),
        autogluon_median_r2=("autogluon_r2", "median"),
        autogluon_wins=("winner_r2", lambda s: (s == "AutoGluon").sum()),
        custom_wins=("winner_r2", lambda s: (s == "Custom").sum()),
        n_targets=("target", "count"),
    )
    .reset_index()
)

comparison_out = OUT / "custom_vs_autogluon_target_level_comparison.csv"
summary_out = OUT / "custom_vs_autogluon_summary.csv"

comparison.to_csv(comparison_out, index=False)
summary.to_csv(summary_out, index=False)

print("\nCUSTOM VS AUTOGLUON TARGET-LEVEL COMPARISON")
print(
    comparison[
        [
            "dataset",
            "target",
            "custom_best_model",
            "custom_r2",
            "autogluon_best_model",
            "autogluon_r2",
            "r2_difference_autogluon_minus_custom",
            "winner_r2",
        ]
    ].to_string(index=False)
)

print("\nSUMMARY")
print(summary.to_string(index=False))

print("\nsaved:", comparison_out)
print("saved:", summary_out)



