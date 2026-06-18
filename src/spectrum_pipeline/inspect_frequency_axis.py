from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


def read_spectrum_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    p = project_root / "data" / "raw" / "spectrum_path.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing spectrum_path.txt at: {p}")
    return Path(p.read_text(encoding="utf-8").strip())


def find_header_line_index(lines: list[str]) -> int | None:
    # the long header line starts with "time [s]" in your file
    for i, line in enumerate(lines):
        if "time [s]" in line and ";" in line:
            return i
    return None


def parse_header_line(header_line: str) -> list[str]:
    parts = [p.strip() for p in header_line.split(";")]
    parts = [p for p in parts if p]  # remove empty from trailing ;
    return parts


def find_first_numeric_row(lines: list[str]) -> int | None:
    num_pattern = re.compile(r"^\s*[-+]?\d+(\.\d*)?([eE][-+]?\d+)?\s*;")
    for i, line in enumerate(lines):
        if num_pattern.match(line):
            return i
    return None


def load_numeric_table(lines: list[str], first_data_idx: int) -> pd.DataFrame:
    numeric_text = "\n".join(lines[first_data_idx:])
    df = pd.read_csv(
        pd.io.common.StringIO(numeric_text),
        sep=";",
        header=None,
        engine="python",
    )
    df = df.dropna(axis=1, how="all")  # remove empty cols from trailing ;
    return df


def main() -> None:
    spectrum_root = read_spectrum_root()
    files = sorted(spectrum_root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {spectrum_root}")

    csv_path = files[0]
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    header_idx = find_header_line_index(lines)
    if header_idx is None:
        raise ValueError("Could not find header line containing 'time [s]'.")

    header_cols = parse_header_line(lines[header_idx])

    first_data_idx = find_first_numeric_row(lines)
    if first_data_idx is None:
        raise ValueError("Could not find first numeric data row.")

    df = load_numeric_table(lines, first_data_idx)

    # Assign column names safely
    if len(header_cols) >= df.shape[1]:
        df.columns = header_cols[: df.shape[1]]
    else:
        df.columns = header_cols + [f"extra_{k}" for k in range(df.shape[1] - len(header_cols))]

    print("Using file:", csv_path.name)
    print("Header columns:", len(header_cols))
    print("Numeric table shape:", df.shape)

    # Find the frequency column
    freq_col = None
    for c in df.columns:
        c_clean = c.replace(" ", "").lower()
        if c_clean.startswith("f[hz]"):
            freq_col = c
            break


    if freq_col is None:
        print("\nCould not find 'f [Hz]' column name in parsed columns.")
        print("Columns containing 'Hz':")
        for c in df.columns:
            if "Hz" in c:
                print(" -", c)
        return

    f = pd.to_numeric(df[freq_col], errors="coerce").dropna()

    print("\nFrequency column:", freq_col)
    print("Count:", len(f))
    print("Min [Hz]:", f.min())
    print("Max [Hz]:", f.max())
    print("\nFirst 10 f values:")
    print(f.head(10).to_string(index=False))
    print("\nLast 10 f values:")
    print(f.tail(10).to_string(index=False))

    # Show which spectrum columns exist (so we know what we can compute bands for)
    print("\nColumns that look like spectra/envelopes (contain 'spec' or 'env'):")
    spec_cols = [c for c in df.columns if ("spec" in c.lower() or "env" in c.lower())]
    print("Count:", len(spec_cols))
    for c in spec_cols[:30]:
        print(" -", c)
    if len(spec_cols) > 30:
        print(" ... (showing first 30)")

if __name__ == "__main__":
    main()


