from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
INPUT_FILE = Path("data/outputs/new_data/comparison/custom_vs_autogluon_target_level_comparison.csv")
OUTPUT_DIR = Path("data/outputs/new_data/comparison/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "target_level_r2_comparison_custom_vs_autogluon.png"
OUT_PDF = OUTPUT_DIR / "target_level_r2_comparison_custom_vs_autogluon.pdf"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)


# ------------------------------------------------------------
# Make clean target labels
# ------------------------------------------------------------
def clean_target_name(target: str) -> str:
    target = target.replace("res_eff_", "")
    target = target.replace("_my_s", "")
    target = target.replace("medium", "mid")
    return target


df["target_clean"] = df["target"].apply(clean_target_name)
df["label"] = df["dataset"] + "\n" + df["target_clean"]


# ------------------------------------------------------------
# Sort targets in a logical order
# ------------------------------------------------------------
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

dataset_order = {"G28": 0, "G200": 1}

df["dataset_order"] = df["dataset"].map(dataset_order)
df["target_order"] = df["target_clean"].map(band_order)

df = df.sort_values(["dataset_order", "target_order"]).reset_index(drop=True)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
x = np.arange(len(df))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 5.5))

ax.bar(x - width / 2, df["custom_r2"], width, label="Custom workflow")
ax.bar(x + width / 2, df["autogluon_r2"], width, label="AutoGluon")

ax.set_ylabel("R²")
ax.set_xlabel("Target variable")
ax.set_title("Target-level R² comparison between custom workflow and AutoGluon")
ax.set_xticks(x)
ax.set_xticklabels(df["label"], rotation=45, ha="right")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

fig.savefig(OUT_PNG, dpi=300)
fig.savefig(OUT_PDF)

print("Saved:")
print(OUT_PNG)
print(OUT_PDF)