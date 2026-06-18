from __future__ import annotations

import re
from pathlib import Path
from io import StringIO
import pandas as pd


def read_spectrum_root() -> Path:
    # project_root = folder that contains "data"
    project_root = Path(__file__).resolve().parents[2]
    p = project_root / "data" / "raw" / "spectrum_path.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing spectrum_path.txt at: {p}")
    return Path(p.read_text(encoding="utf-8").strip())


def find_header_line_index(lines: list[str]) -> int | None:
    """
    Find the line that contains the long header with many ';' fields,
    usually starting with 'time [s]'.
    """
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("time") and ";".join([""]) is not None and ";" in s:
            # extra check: must contain frequency column too
            if "f [hz]" in s or "f[hz]" in s:
                return i
    return None


def parse_header_line(header_line: str) -> list[str]:
    """
    Split header by ';' and clean tokens.
    Example token: 'angleOR [rad]' or 'specX[um/s]'
    """
    raw = [t.strip() for t in header_line.split(";")]
    # remove empty tokens (from trailing semicolons)
    raw = [t for t in raw if t != ""]
    return raw


def find_first_numeric_row(lines: list[str]) -> int | None:
    # numeric row usually looks like: 0.0000e+00 ; 0.0000e+00 ; ...
    num_pattern = re.compile(r"^\s*[-+]?\d+(\.\d*)?([eE][-+]?\d+)?\s*;")
    for i, line in enumerate(lines):
        if num_pattern.match(line):
            return i
    return None
def find_header_line_index(lines: list[str]) -> int | None:
    # The header line contains "time [s]" AND "f[Hz]" in these files
    for i, line in enumerate(lines):
        if "time [s]" in line and "f[Hz]" in line:
            return i
    return None


def parse_header_columns(lines: list[str]) -> list[str]:
    idx = find_header_line_index(lines)
    if idx is None:
        raise ValueError("Could not find header line (time [s] ... f[Hz] ...).")

    header_line = lines[idx].strip()

    # split by ';' and clean
    parts = [p.strip() for p in header_line.split(";")]
    parts = [p for p in parts if p != ""]  # remove empty parts

    return parts


def load_numeric_table(lines: list[str], first_data_idx: int, col_names: list[str]) -> pd.DataFrame:
    numeric_text = "\n".join(lines[first_data_idx:])
    df = pd.read_csv(
        StringIO(numeric_text),
        sep=";",
        header=None,
        engine="python",
    )
    # drop fully-empty columns (caused by trailing ';')
    df = df.dropna(axis=1, how="all")
    # --- Assign column names (as many as the numeric table actually has) ---
    available = min(len(col_names), df.shape[1])
    df.columns = col_names[:available]

    if len(col_names) != df.shape[1]:
        print(f"WARNING: header has {len(col_names)} columns, numeric table has {df.shape[1]} columns.")
        if len(col_names) > df.shape[1]:
            missing = col_names[df.shape[1]:]
            print("Columns present in header but missing in numeric data (likely empty at end):")
            for name in missing:
                print("  -", name)

    return df


def main() -> None:
    spectrum_root = read_spectrum_root()
    files = sorted(spectrum_root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {spectrum_root}")

    csv_path = files[0]
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    col_names = parse_header_columns(lines)


    header_idx = find_header_line_index(lines)
    if header_idx is None:
        raise ValueError("Could not find the long header line (starting with 'time [s]' ...).")

    header_cols = parse_header_line(lines[header_idx])

    first_data_idx = find_first_numeric_row(lines)
    if first_data_idx is None:
        raise ValueError("Could not find first numeric data row in the file.")

    df = load_numeric_table(lines, first_data_idx, col_names)


    # Assign names safely (match the numeric columns count)
    if len(header_cols) >= df.shape[1]:
        df.columns = header_cols[: df.shape[1]]
    else:
        # fallback: keep numeric columns for the extra ones
        df.columns = header_cols + [f"extra_{k}" for k in range(df.shape[1] - len(header_cols))]

    print("Using file:", csv_path.name)
    print("Header line index (0-based):", header_idx)
    print("First numeric row index (0-based):", first_data_idx)
    print("Parsed header columns:", len(header_cols))
    print("Numeric DataFrame shape:", df.shape)
    print("\nFirst 3 rows (with column names):")
    print(df.head(3).to_string(index=False))

    print("\nColumn names (first 20):")
    for c in df.columns[:20]:
        print(" -", c)


if __name__ == "__main__":
    main()


