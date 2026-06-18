from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE = Path("data/outputs/new_data")

CUSTOM_PRED = BASE / "custom_workflow" / "predictions"
AG_PRED = BASE / "autogluon_fixed_split" / "predictions"
COMPARISON = BASE / "comparison" / "custom_vs_autogluon_target_level_comparison.csv"

PLOT_OUT = BASE / "comparison" / "plots"
PLOT_OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Thesis-relevant representative targets for case-level plots
# ------------------------------------------------------------
PLOT_TARGETS = [
    ("G28", "res_eff_broad_y_my_s"),
    ("G28", "res_eff_broad_z_my_s"),
    ("G200", "res_eff_broad_y_my_s"),
    ("G200", "res_eff_broad_z_my_s"),
    ("G200", "res_eff_low_y_my_s"),
]


# ------------------------------------------------------------
# Load comparison file
# ------------------------------------------------------------
comparison = pd.read_csv(COMPARISON)

plot_summary_rows = []


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def safe_name(text):
    return text.replace("/", "_").replace("\\", "_").replace(":", "_")


def clean_target_name(target: str) -> str:
    """
    Convert long result column names into short readable labels.
    Example:
    res_eff_broad_y_my_s -> broad_y
    res_eff_medium_z_my_s -> mid_z
    """
    target = target.replace("res_eff_", "")
    target = target.replace("_my_s", "")
    target = target.replace("medium", "mid")
    return target


# ------------------------------------------------------------
# Existing plots: predicted vs simulated, residuals, error distribution
# ------------------------------------------------------------
for dataset, target in PLOT_TARGETS:
    print("\nCreating plots:", dataset, target)

    row = comparison[
        (comparison["dataset"] == dataset) &
        (comparison["target"] == target)
    ].iloc[0]

    custom_model = row["custom_best_model"]

    custom_file = CUSTOM_PRED / f"custom_predictions_{dataset}_{target}.csv"
    ag_file = AG_PRED / f"autogluon_predictions_{dataset}_{target}.csv"

    custom_df = pd.read_csv(custom_file)
    ag_df = pd.read_csv(ag_file)

    y_true = custom_df["y_true"]
    pred_custom = custom_df[f"pred_{custom_model}"]
    pred_ag = ag_df["pred_autogluon"]

    residual_custom = y_true - pred_custom
    residual_ag = y_true - pred_ag

    abs_error_custom = residual_custom.abs()
    abs_error_ag = residual_ag.abs()

    name = safe_name(f"{dataset}_{target}")

    # ------------------------------------------------------------
    # Plot 1: Predicted versus simulated
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 6))

    plt.scatter(y_true, pred_custom, alpha=0.7, label=f"Custom ({custom_model})")
    plt.scatter(y_true, pred_ag, alpha=0.7, marker="x", label="AutoGluon")

    min_val = min(y_true.min(), pred_custom.min(), pred_ag.min())
    max_val = max(y_true.max(), pred_custom.max(), pred_ag.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Ideal")

    plt.xlabel("Simulated effective value in m/s")
    plt.ylabel("Predicted effective value in m/s")
    plt.title(f"Predicted versus simulated\n{dataset} - {clean_target_name(target)}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_OUT / f"{name}_predicted_vs_simulated.png", dpi=300)
    plt.savefig(PLOT_OUT / f"{name}_predicted_vs_simulated.pdf")
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: Residuals
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.scatter(y_true, residual_custom, alpha=0.7, label=f"Custom ({custom_model})")
    plt.scatter(y_true, residual_ag, alpha=0.7, marker="x", label="AutoGluon")
    plt.axhline(0, linestyle="--")

    plt.xlabel("Simulated effective value in m/s")
    plt.ylabel("Residual in m/s")
    plt.title(f"Residual comparison\n{dataset} - {clean_target_name(target)}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_OUT / f"{name}_residuals.png", dpi=300)
    plt.savefig(PLOT_OUT / f"{name}_residuals.pdf")
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: Absolute error distribution
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 5))

    plt.boxplot(
        [abs_error_custom, abs_error_ag],
        labels=[f"Custom\n{custom_model}", "AutoGluon"],
    )

    plt.ylabel("Absolute error in m/s")
    plt.title(f"Absolute error distribution\n{dataset} - {clean_target_name(target)}")
    plt.tight_layout()

    plt.savefig(PLOT_OUT / f"{name}_absolute_error_distribution.png", dpi=300)
    plt.savefig(PLOT_OUT / f"{name}_absolute_error_distribution.pdf")
    plt.close()

    # ------------------------------------------------------------
    # Save per-case error table
    # ------------------------------------------------------------
    error_df = pd.DataFrame({
        "dataset": dataset,
        "target": target,
        "y_true": y_true,
        "custom_model": custom_model,
        "pred_custom": pred_custom,
        "pred_autogluon": pred_ag,
        "residual_custom": residual_custom,
        "residual_autogluon": residual_ag,
        "abs_error_custom": abs_error_custom,
        "abs_error_autogluon": abs_error_ag,
    })

    error_df.to_csv(PLOT_OUT / f"{name}_case_errors.csv", index=False)

    plot_summary_rows.append({
        "dataset": dataset,
        "target": target,
        "target_clean": clean_target_name(target),
        "custom_model": custom_model,
        "custom_r2": r2_score(y_true, pred_custom),
        "autogluon_r2": r2_score(y_true, pred_ag),
        "custom_mae": mean_absolute_error(y_true, pred_custom),
        "autogluon_mae": mean_absolute_error(y_true, pred_ag),
        "custom_rmse": rmse(y_true, pred_custom),
        "autogluon_rmse": rmse(y_true, pred_ag),
        "plot_predicted_vs_simulated": str(PLOT_OUT / f"{name}_predicted_vs_simulated.png"),
        "plot_residuals": str(PLOT_OUT / f"{name}_residuals.png"),
        "plot_absolute_error_distribution": str(PLOT_OUT / f"{name}_absolute_error_distribution.png"),
    })


# ------------------------------------------------------------
# Save plot summary table
# ------------------------------------------------------------
summary = pd.DataFrame(plot_summary_rows)
summary_out = PLOT_OUT / "plot_summary.csv"
summary.to_csv(summary_out, index=False)


# ------------------------------------------------------------
# New plot: target-level R² comparison for all 16 targets
# ------------------------------------------------------------
target_r2 = comparison.copy()

target_r2["target_clean"] = target_r2["target"].apply(clean_target_name)

band_order = {
    "broad_y": 0,
    "broad_z": 1,
    "low_y": 2,
    "low_z": 3,
    "mid_y": 4,
    "mid_z": 5,
    "high_y": 6,
    "high_z": 7,
}

dataset_order = {
    "G28": 0,
    "G200": 1,
}

target_r2["dataset_order"] = target_r2["dataset"].map(dataset_order)
target_r2["target_order"] = target_r2["target_clean"].map(band_order)

target_r2 = target_r2.sort_values(
    ["dataset_order", "target_order"]
).reset_index(drop=True)

x = np.arange(len(target_r2))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 5.5))

ax.bar(
    x - width / 2,
    target_r2["custom_r2"],
    width,
    label="Custom workflow",
)

ax.bar(
    x + width / 2,
    target_r2["autogluon_r2"],
    width,
    label="AutoGluon",
)

# Short labels only on x-axis
ax.set_xticks(x)
ax.set_xticklabels(target_r2["target_clean"], rotation=45, ha="right")

# Clear separation between G28 and G200
ax.axvline(7.5, linestyle="--", linewidth=1)

ax.text(3.5, 1.02, "G28", ha="center", va="bottom", fontsize=11)
ax.text(11.5, 1.02, "G200", ha="center", va="bottom", fontsize=11)

ax.set_ylabel("R²")
ax.set_xlabel("Target variable")
ax.set_title("Target-level R² comparison between custom workflow and AutoGluon")
ax.set_ylim(0, 1.08)
ax.legend(loc="lower left")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

target_r2_png = PLOT_OUT / "target_level_r2_comparison_custom_vs_autogluon_grouped.png"
target_r2_pdf = PLOT_OUT / "target_level_r2_comparison_custom_vs_autogluon_grouped.pdf"

fig.savefig(target_r2_png, dpi=300)
fig.savefig(target_r2_pdf)
plt.close(fig)


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------
print("\nDONE PLOTS")
print(summary[[
    "dataset",
    "target",
    "custom_model",
    "custom_r2",
    "autogluon_r2",
    "custom_mae",
    "autogluon_mae",
]].to_string(index=False))

print("\nsaved plots in:", PLOT_OUT)
print("saved:", summary_out)

print("\nSaved target-level R² comparison:")
print(target_r2_png)
print(target_r2_pdf)