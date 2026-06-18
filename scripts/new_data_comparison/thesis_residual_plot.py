from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("data/outputs/new_data")
CUSTOM_PRED = BASE / "custom_workflow" / "predictions"
AG_PRED = BASE / "autogluon_fixed_split" / "predictions"
OUT = BASE / "comparison" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

dataset = "G200"
target = "res_eff_broad_y_my_s"
custom_model = "SVR"

custom = pd.read_csv(CUSTOM_PRED / f"custom_predictions_{dataset}_{target}.csv")
ag = pd.read_csv(AG_PRED / f"autogluon_predictions_{dataset}_{target}.csv")

y_true = custom["y_true"]
res_custom = y_true - custom[f"pred_{custom_model}"]
res_ag = y_true - ag["pred_autogluon"]

plt.figure(figsize=(6.2, 4.0))
plt.scatter(y_true, res_custom, label="Custom workflow", marker="o")
plt.scatter(y_true, res_ag, label="AutoGluon", marker="x")
plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel(r"Simulated effective value $v_\mathrm{eff}$ in m/s")
plt.ylabel(r"Residual $e$ in m/s")
plt.title("Residual plot for G200 broad-band Y target")
plt.legend()
plt.tight_layout()

out = OUT / "G200_res_eff_broad_y_my_s_residual_plot_thesis.png"
plt.savefig(out, dpi=300)
plt.close()

print("DONE residual plot")
print("saved:", out)



