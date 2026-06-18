import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("./data/outputs/l_effective")
rng = np.random.default_rng(42)

Xtr = pd.read_csv(BASE / "X_train.csv")
Xva = pd.read_csv(BASE / "X_val.csv")
Xte = pd.read_csv(BASE / "X_test.csv")
Xall = pd.concat([Xtr, Xva, Xte], axis=0).reset_index(drop=True)

centers = pd.read_csv(BASE / "adaptive_full_input_new_simulation_proposals.csv")

INPUTS = list(Xtr.columns)
N_PER_REGION = 5

mins = Xall[INPUTS].min()
maxs = Xall[INPUTS].max()
stds = Xall[INPUTS].std()

rows = []

for i, center in centers.iterrows():
    for j in range(N_PER_REGION):
        row = {
            "proposal_id": f"{center['target_name']}_cluster{int(center['cluster'])}_p{j+1}",
            "source_target": center["target_name"],
            "source_cluster": int(center["cluster"]),
            "representative_adaptive_score": center["representative_adaptive_score"],
        }

        for col in INPUTS:
            base_val = float(center[col])
            noise_scale = 0.05 * float(stds[col])

            if j == 0:
                # keep one exact representative point
                new_val = base_val
            else:
                new_val = base_val + rng.normal(0.0, noise_scale)

            new_val = float(np.clip(new_val, mins[col], maxs[col]))

            # Z is integer-like
            if col == "Z":
                new_val = int(round(new_val))

            row[col] = new_val

        rows.append(row)

df = pd.DataFrame(rows)

# remove exact duplicate input rows if any
df = df.drop_duplicates(subset=INPUTS).reset_index(drop=True)

out = BASE / "adaptive_enrichment_100_new_simulation_points.csv"
df.to_csv(out, index=False)

print("saved:", out)
print("shape:", df.shape)
print(df.head(20).to_string(index=False))

