from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

BASE = Path("data/outputs/new_data")
CUSTOM_PRED = BASE / "custom_workflow" / "predictions"
AG_PRED = BASE / "autogluon_fixed_split" / "predictions"
OUT = BASE / "comparison" / "plots" / "thesis_final"
OUT.mkdir(parents=True, exist_ok=True)

plots = [
    ("G28", "res_eff_broad_y_my_s", "G28_broad_Y_predicted_vs_simulated_thesis.png"),
    ("G28", "res_eff_broad_z_my_s", "G28_broad_Z_predicted_vs_simulated_thesis.png"),
    ("G200", "res_eff_broad_y_my_s", "G200_broad_Y_predicted_vs_simulated_thesis.png"),
    ("G200", "res_eff_broad_z_my_s", "G200_broad_Z_predicted_vs_simulated_thesis.png"),
]

for dataset, target, filename in plots:
    custom = pd.read_csv(CUSTOM_PRED / f"custom_predictions_{dataset}_{target}.csv")
    ag = pd.read_csv(AG_PRED / f"autogluon_predictions_{dataset}_{target}.csv")

    y_true = custom["y_true"]
    pred_custom = custom["pred_SVR"]
    pred_ag = ag["pred_autogluon"]

    min_val = min(y_true.min(), pred_custom.min(), pred_ag.min())
    max_val = max(y_true.max(), pred_custom.max(), pred_ag.max())

    plt.figure(figsize=(6.2, 4.0))
    plt.scatter(y_true, pred_custom, label="Custom workflow", marker="o", s=22)
    plt.scatter(y_true, pred_ag, label="AutoGluon", marker="x", s=28)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1, label="Ideal")

    plt.xlabel(r"Simulated effective value $v_\mathrm{eff}$ in m/s")
    plt.ylabel(r"Predicted effective value $\hat{v}_\mathrm{eff}$ in m/s")
    plt.legend()
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()

    out = OUT / filename
    plt.savefig(out, dpi=300)
    plt.close()

    print("saved:", out)

print("\nDONE thesis prediction plots")



