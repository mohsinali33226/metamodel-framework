from pathlib import Path
import pandas as pd

BASE = Path("data/outputs/new_data")
CUSTOM_PRED = BASE / "custom_workflow" / "predictions"
AG_PRED = BASE / "autogluon_fixed_split" / "predictions"
COMPARISON = BASE / "comparison" / "custom_vs_autogluon_target_level_comparison.csv"

OUT = BASE / "comparison" / "error_analysis"
OUT.mkdir(parents=True, exist_ok=True)

comparison = pd.read_csv(COMPARISON)

summary_rows = []
worst_rows = []

for _, row in comparison.iterrows():
    dataset = row["dataset"]
    target = row["target"]
    custom_model = row["custom_best_model"]

    custom_file = CUSTOM_PRED / f"custom_predictions_{dataset}_{target}.csv"
    ag_file = AG_PRED / f"autogluon_predictions_{dataset}_{target}.csv"

    custom_df = pd.read_csv(custom_file)
    ag_df = pd.read_csv(ag_file)

    y_true = custom_df["y_true"]
    pred_custom = custom_df[f"pred_{custom_model}"]
    pred_ag = ag_df["pred_autogluon"]

    errors = pd.DataFrame({
        "dataset": dataset,
        "target": target,
        "case_index": range(len(y_true)),
        "y_true": y_true,
        "custom_model": custom_model,
        "pred_custom": pred_custom,
        "pred_autogluon": pred_ag,
    })

    errors["residual_custom"] = errors["y_true"] - errors["pred_custom"]
    errors["residual_autogluon"] = errors["y_true"] - errors["pred_autogluon"]

    errors["abs_error_custom"] = errors["residual_custom"].abs()
    errors["abs_error_autogluon"] = errors["residual_autogluon"].abs()

    errors["better_method_case"] = errors.apply(
        lambda r: "AutoGluon" if r["abs_error_autogluon"] < r["abs_error_custom"] else "Custom",
        axis=1,
    )

    errors.to_csv(OUT / f"case_errors_{dataset}_{target}.csv", index=False)

    summary_rows.append({
        "dataset": dataset,
        "target": target,
        "custom_model": custom_model,
        "mean_residual_custom": errors["residual_custom"].mean(),
        "mean_residual_autogluon": errors["residual_autogluon"].mean(),
        "mean_abs_error_custom": errors["abs_error_custom"].mean(),
        "mean_abs_error_autogluon": errors["abs_error_autogluon"].mean(),
        "median_abs_error_custom": errors["abs_error_custom"].median(),
        "median_abs_error_autogluon": errors["abs_error_autogluon"].median(),
        "max_abs_error_custom": errors["abs_error_custom"].max(),
        "max_abs_error_autogluon": errors["abs_error_autogluon"].max(),
        "autogluon_better_cases": (errors["better_method_case"] == "AutoGluon").sum(),
        "custom_better_cases": (errors["better_method_case"] == "Custom").sum(),
        "n_cases": len(errors),
    })

    worst_custom = errors.sort_values("abs_error_custom", ascending=False).head(10).copy()
    worst_custom["method"] = "Custom"

    worst_ag = errors.sort_values("abs_error_autogluon", ascending=False).head(10).copy()
    worst_ag["method"] = "AutoGluon"

    worst_rows.append(worst_custom)
    worst_rows.append(worst_ag)

summary = pd.DataFrame(summary_rows)
worst = pd.concat(worst_rows, axis=0).reset_index(drop=True)

summary_out = OUT / "error_summary_by_target.csv"
worst_out = OUT / "worst_10_errors_by_target_and_method.csv"

summary.to_csv(summary_out, index=False)
worst.to_csv(worst_out, index=False)

print("\nDONE ERROR ANALYSIS")
print(summary[[
    "dataset",
    "target",
    "mean_abs_error_custom",
    "mean_abs_error_autogluon",
    "max_abs_error_custom",
    "max_abs_error_autogluon",
    "autogluon_better_cases",
    "custom_better_cases",
]].to_string(index=False))

print("\nsaved:", summary_out)
print("saved:", worst_out)
print("saved detailed case files in:", OUT)



