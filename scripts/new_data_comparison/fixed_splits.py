from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/new_data")
OUT = Path("data/outputs/new_data/fixed_splits")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "G28": RAW / "G28" / "6206_G28_Welligkeit=keine_Datenbank_erweitert.csv",
    "G200": RAW / "G200" / "6206_G200_Welligkeit=keine_Datenbank_erweitert.csv",
}

TARGETS_YZ = [
    "res_eff_broad_y_my_s",
    "res_eff_broad_z_my_s",
    "res_eff_low_y_my_s",
    "res_eff_low_z_my_s",
    "res_eff_medium_y_my_s",
    "res_eff_medium_z_my_s",
    "res_eff_high_y_my_s",
    "res_eff_high_z_my_s",
]

def load_csv(path):
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = [c.replace("\ufeff", "") for c in df.columns]
    return df

for dataset_name, path in FILES.items():
    df = load_csv(path)

    input_cols = [
        c for c in df.columns
        if not c.startswith("res_") and c != "id"
    ]

    keep_cols = input_cols + TARGETS_YZ
    df = df[keep_cols].copy()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    train_path = OUT / f"{dataset_name}_train.csv"
    test_path = OUT / f"{dataset_name}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"{dataset_name}")
    print("train:", train_df.shape, "->", train_path)
    print("test: ", test_df.shape, "->", test_path)
    print("input columns:", len(input_cols))
    print("targets:", len(TARGETS_YZ))
    print()

print("PHASE 3 SPLIT FILES CREATED")



