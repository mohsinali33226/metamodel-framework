from __future__ import annotations

import sys
from pathlib import Path
import argparse

import pandas as pd

# ------------------------------------------------------------
# Make project root + src importable
# ------------------------------------------------------------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]  # .../metamodel_framework
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from framework.utils.paths import load_output_paths


def main():
    ap = argparse.ArgumentParser(description="Build spectrum dataset by merging sampling inputs with band RMS outputs.")
    ap.add_argument(
        "--sampling_csv",
        default=str(ROOT / "data" / "raw" / "sampling" / "sampling.csv"),
        help="Path to sampling.csv (inputs/features). Default: data/raw/sampling/sampling.csv",
    )
    ap.add_argument(
        "--join",
        default="file_name",
        choices=["file_name", "row_order"],
        help="How to merge. 'file_name' (default) joins on file_name column. "
             "'row_order' merges by sorted order (fallback).",
    )
    args = ap.parse_args()

    sampling_path = Path(args.sampling_csv)
    if not sampling_path.exists():
        raise FileNotFoundError(f"sampling_csv not found: {sampling_path}")

    cfg = load_output_paths()
    sp = cfg["spectrum"]

    band_rms_path = Path(sp["band_rms_all_files"])
    out_dataset_path = Path(sp["dataset"])

    if not band_rms_path.exists():
        raise FileNotFoundError(
            f"Missing band RMS file: {band_rms_path}\n"
            f"Run compute_band_rms_all.py first."
        )

    sampling_df = pd.read_csv(sampling_path)
    band_df = pd.read_csv(band_rms_path)

    # Remove t_s if present (not useful for generalization)
    if "t_s" in sampling_df.columns:
        sampling_df = sampling_df.drop(columns=["t_s"])
        print("Dropped column: t_s from sampling_df")

    # We expect band_df to have file_name
    if args.join == "file_name":
        if "file_name" not in band_df.columns:
            raise ValueError("band_rms_all_files.csv does not contain 'file_name' column.")
        if "file_name" not in sampling_df.columns:
            # if sampling doesn't have file_name, we cannot do this join
            raise ValueError(
                "sampling.csv does not contain 'file_name'. "
                "Use --join row_order or add file_name to sampling table."
            )

        merged = pd.merge(sampling_df, band_df, on="file_name", how="inner")

        if len(merged) == 0:
            raise ValueError(
                "Merge produced 0 rows. Check that sampling.csv file_name values match band_rms_all_files.csv."
            )

        print(f"Merged on file_name: rows={len(merged)}")

    else:
        # fallback: join by sorted order
        sampling_sorted = sampling_df.sort_index().reset_index(drop=True)
        band_sorted = band_df.sort_values("file_name").reset_index(drop=True) if "file_name" in band_df.columns else band_df.reset_index(drop=True)

        n = min(len(sampling_sorted), len(band_sorted))
        if n == 0:
            raise ValueError("One of the inputs is empty. Cannot merge by row_order.")

        merged = pd.concat([sampling_sorted.iloc[:n, :], band_sorted.iloc[:n, :]], axis=1)
        print(f"Merged by row_order: rows={len(merged)} (min of both)")

    out_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dataset_path, index=False)

    print(f"Saved spectrum dataset: {out_dataset_path}")
    print(f"Shape: {merged.shape}")


if __name__ == "__main__":
    main()

