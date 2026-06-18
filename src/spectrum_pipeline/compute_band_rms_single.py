from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# We reuse the same robust parsing approach you already built
# (make sure this file exists and works: src/extract_one_spectrum_table.py)
from extract_spectrum_table import (
    find_first_numeric_row,
    find_header_line_index,
    parse_header_line,
    read_spectrum_root,
    load_numeric_table,
)


# ---------------------------
# 1) Configure frequency bands here
# ---------------------------
# You can edit these later once you confirm the exact DIN ISO 15242 band limits.
BANDS_HZ: Dict[str, Tuple[float, float]] = {
    "low": (0.0, 300.0),
    "mid": (300.0, 3000.0),
    "high": (3000.0, 12000.0),
    "broad": (0.0, 12000.0),
}


@dataclass
class BandResult:
    signal: str
    band: str
    f_min: float
    f_max: float
    rms: float


def _find_frequency_column(columns: List[str]) -> str:
    """
    Robustly detect the frequency column.
    Typical case in your files: 'f[Hz]'
    """
    # exact best match first
    for c in columns:
        if c.strip().lower() in {"f[hz]", "f [hz]"}:
            return c

    # fallback: contains "hz" and starts with f
    candidates = [c for c in columns if ("hz" in c.lower()) and c.strip().lower().startswith("f")]
    if candidates:
        return candidates[0]

    # fallback: any column containing "hz"
    candidates = [c for c in columns if "hz" in c.lower()]
    if candidates:
        return candidates[0]

    raise ValueError("Could not detect frequency column (no column contains 'Hz').")


def _find_signal_columns(columns: List[str]) -> List[str]:
    """
    Heuristic: spectrum/envelope signals contain 'spec' or 'env'
    """
    signals = [c for c in columns if ("spec" in c.lower()) or ("env" in c.lower())]
    # remove obvious non-signal columns if any appear
    return signals


def band_rms(f: np.ndarray, y: np.ndarray, f_min: float, f_max: float) -> float:
    """
    RMS over a frequency band [f_min, f_max] using trapezoidal integration.

    RMS = sqrt( (1/(f_max-f_min)) * integral_{f_min}^{f_max} y(f)^2 df )
    """
    if f.size == 0 or y.size == 0:
        return float("nan")

    # Ensure arrays are 1D float
    f = np.asarray(f, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    mask = (f >= f_min) & (f <= f_max)
    f_band = f[mask]
    y_band = y[mask]

    if f_band.size < 2:
        return float("nan")

    # IMPORTANT: NumPy 2.x => use trapezoid(), NOT trapz()
    integral = np.trapezoid(y_band ** 2, f_band)
    width = float(f_max - f_min)
    if width <= 0:
        return float("nan")

    return float(np.sqrt(integral / width))


def load_one_spectrum_csv(csv_path: Path) -> pd.DataFrame:
    """
    Reads a Bearinx output CSV with messy header + numeric table.
    Uses your helper functions from extract_one_spectrum_table.py
    """
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    header_idx = find_header_line_index(lines)
    if header_idx is None:
        raise ValueError("Could not find the long header line (starting with 'time [s]'...).")

    header_cols = parse_header_line(lines[header_idx])

    first_data_idx = find_first_numeric_row(lines)
    if first_data_idx is None:
        raise ValueError("Could not find the first numeric data row in the file.")

    # load_numeric_table() returns numeric df with raw columns count
    df = load_numeric_table(lines, first_data_idx, header_cols)

    # Assign names safely (match numeric columns count)
    if len(header_cols) >= df.shape[1]:
        df.columns = header_cols[: df.shape[1]]
    else:
        df.columns = header_cols + [f"extra_{k}" for k in range(df.shape[1] - len(header_cols))]

    return df


def main() -> None:
    spectrum_root = read_spectrum_root()
    files = sorted(Path(spectrum_root).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {spectrum_root}")

    csv_path = files[0]  # first file for now
    print("Using file:", csv_path.name)

    df = load_one_spectrum_csv(csv_path)

    # Detect frequency + signals
    freq_col = _find_frequency_column(list(df.columns))
    signals = _find_signal_columns(list(df.columns))

    print("Frequency column:", freq_col)
    print("Signals found:", len(signals))

    f = pd.to_numeric(df[freq_col], errors="coerce").to_numpy()
    # remove NaNs from frequency (and align signals later)
    valid_f = np.isfinite(f)
    f = f[valid_f]

    results: List[BandResult] = []

    for sig in signals:
        y = pd.to_numeric(df[sig], errors="coerce").to_numpy()
        y = y[valid_f]  # align with filtered frequency

        for band_name, (f_min, f_max) in BANDS_HZ.items():
            rms_val = band_rms(f, y, f_min, f_max)
            results.append(BandResult(sig, band_name, f_min, f_max, rms_val))

    out_df = pd.DataFrame([r.__dict__ for r in results])

    # Print a small preview
    print("\n--- Band RMS preview (first 20 rows) ---")
    print(out_df.head(20).to_string(index=False))

    # Save output CSV
    out_dir = Path("data") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"band_rms__{csv_path.stem}.csv"
    out_df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv.as_posix())


if __name__ == "__main__":
    main()


