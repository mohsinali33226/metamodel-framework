from __future__ import annotations

import sys
from pathlib import Path
import argparse

import numpy as np
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
from extract_spectrum_table import (
    find_first_numeric_row,
    find_header_line_index,
    parse_header_line,
    read_spectrum_root,
    load_numeric_table,
)


BANDS_HZ = {
    "low": (0.0, 300.0),
    "mid": (300.0, 3000.0),
    "high": (3000.0, 12000.0),
    "broad": (0.0, 12000.0),
}


def band_rms(f: np.ndarray, y: np.ndarray, f_min: float, f_max: float) -> float:
    if f.size == 0 or y.size == 0:
        return float("nan")

    f = np.asarray(f, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    mask = (f >= f_min) & (f <= f_max)
    f_band = f[mask]
    y_band = y[mask]

    if f_band.size < 2:
        return float("nan")

    integral = np.trapezoid(y_band ** 2, f_band)
    width = float(f_max - f_min)
    if width <= 0:
        return float("nan")

    return float(np.sqrt(integral / width))


def load_one_spectrum_csv(csv_path: Path) -> pd.DataFrame:
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    header_idx = find_header_line_index(lines)
    if header_idx is None:
        raise ValueError("Could not find header line with 'time [s]' and 'f[Hz]'.")

    header_cols = parse_header_line(lines[header_idx])

    first_data_idx = find_first_numeric_row(lines)
    if first_data_idx is None:
        raise ValueError("Could not find first numeric data row.")

    df = load_numeric_table(lines, first_data_idx, header_cols)

    if len(header_cols) >= df.shape[1]:
        df.columns = header_cols[: df.shape[1]]
    else:
        df.columns = header_cols + [f"extra_{k}" for k in range(df.shape[1] - len(header_cols))]

    return df


def _find_frequency_column(columns: list[str]) -> str:
    for c in columns:
        s = c.strip().lower()
        if s in {"f[hz]", "f [hz]"}:
            return c

    for c in columns:
        s = c.strip().lower()
        if "hz" in s and s.startswith("f"):
            return c

    for c in columns:
        if "hz" in c.lower():
            return c

    raise ValueError("Could not detect frequency column.")


def _find_signal_columns(columns: list[str]) -> list[str]:
    keep = []
    for c in columns:
        s = c.lower()
        if ("spec" in s or "env" in s) and ("hz" not in s):
            keep.append(c)
    return keep


def compute_band_rms_one_file(df: pd.DataFrame) -> dict:
    freq_col = _find_frequency_column(list(df.columns))
    signal_cols = _find_signal_columns(list(df.columns))

    f = pd.to_numeric(df[freq_col], errors="coerce").to_numpy()
    valid_f = np.isfinite(f)
    f = f[valid_f]

    out = {}

    for sig in signal_cols:
        y = pd.to_numeric(df[sig], errors="coerce").to_numpy()
        y = y[valid_f]

        for band_name, (f_min, f_max) in BANDS_HZ.items():
            out[f"{sig}__{band_name}"] = band_rms(f, y, f_min, f_max)

    return out


def main():
    ap = argparse.ArgumentParser(description="Compute band RMS features for all spectrum CSV files.")
    ap.add_argument(
        "--spectrum_dir",
        default=None,
        help="Folder containing spectrum CSV files. If omitted, read from data/raw/spectrum_path.txt",
    )
    ap.add_argument("--pattern", default="*.csv", help="Glob pattern, default '*.csv'")
    args = ap.parse_args()

    if args.spectrum_dir:
        spectrum_dir = Path(args.spectrum_dir)
    else:
        spectrum_dir = read_spectrum_root()

    if not spectrum_dir.exists():
        raise FileNotFoundError(f"spectrum_dir does not exist: {spectrum_dir}")

    cfg = load_output_paths()
    sp = cfg["spectrum"]
    out_path = Path(sp["band_rms_all_files"])

    files = sorted(spectrum_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {spectrum_dir}")

    rows = []
    for i, fp in enumerate(files, start=1):
        try:
            df = load_one_spectrum_csv(fp)
            feats = compute_band_rms_one_file(df)
            feats["file_name"] = fp.name
            rows.append(feats)
        except Exception as e:
            rows.append({"file_name": fp.name, "error": str(e)})

        if i % 200 == 0:
            print(f"Processed {i}/{len(files)}")

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved band RMS features: {out_path}")
    print(f"Shape: {out_df.shape}")


if __name__ == "__main__":
    main()

