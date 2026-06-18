from pathlib import Path
import pandas as pd

BASE = Path("data/outputs/new_data")
OUT = BASE / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

important_paths = [
    BASE / "fixed_splits",
    BASE / "custom_workflow",
    BASE / "autogluon_fixed_split",
    BASE / "comparison",
    BASE / "comparison" / "plots",
    BASE / "comparison" / "error_analysis",
]

rows = []

for folder in important_paths:
    if folder.exists():
        for p in folder.rglob("*"):
            if p.is_file():
                rows.append({
                    "file": str(p),
                    "folder": str(p.parent),
                    "name": p.name,
                    "suffix": p.suffix,
                    "size_kb": round(p.stat().st_size / 1024, 2),
                })

inventory = pd.DataFrame(rows).sort_values(["folder", "name"])

out_file = OUT / "output_inventory.csv"
inventory.to_csv(out_file, index=False)

print("\nDONE OUTPUT INVENTORY")
print("files listed:", len(inventory))
print("saved:", out_file)
print("\nMain folders checked:")
for folder in important_paths:
    print(folder)



