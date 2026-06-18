from pathlib import Path

# 1) Read the path to the big spectrum folder from your text file
path_file = Path("data/raw/spectrum_path.txt")
spectrum_root = Path(path_file.read_text(encoding="utf-8").strip())

print("Spectrum root:", spectrum_root)
print("Exists:", spectrum_root.exists())

# 2) Pick ONE csv file (the first one alphabetically)
csv_files = sorted(spectrum_root.glob("*.csv"))
print("Number of spectrum CSVs found:", len(csv_files))

if not csv_files:
    raise SystemExit("No CSV files found in spectrum_root. Check spectrum_path.txt")

one_file = csv_files[0]
print("Example file:", one_file.name)

# 3) Read as plain text lines and find where the numeric tables begin
lines = one_file.read_text(encoding="utf-8", errors="ignore").splitlines()
print("Total lines:", len(lines))

def find_line_index(prefix: str):
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            return i
    return None

idx_time = find_line_index("time [s]")
idx_freq = find_line_index("f [Hz]")

print("Line index where time-table header starts:", idx_time)
print("Line index where frequency-table header starts:", idx_freq)

# Print a few lines around what we found (for sanity)
if idx_time is not None:
    print("\n--- Around time-table header ---")
    for j in range(max(0, idx_time-2), min(len(lines), idx_time+3)):
        print(f"{j:4d}: {lines[j][:120]}")

if idx_freq is not None:
    print("\n--- Around frequency-table header ---")
    for j in range(max(0, idx_freq-2), min(len(lines), idx_freq+3)):
        print(f"{j:4d}: {lines[j][:120]}")

    # Print the full frequency-table header line (the one that contains "f [Hz]")
    header_line = lines[idx_freq].strip()
    print("\n--- Full frequency-table header line ---")
    print(header_line)
    # --- Search for "effective value" related keywords (to locate band outputs) ---
keywords = [
    "effective", "eff", "rms",
    "broad", "broadband",
    "low", "mid", "middle", "high",
    "wert", "effektiv", "band"
]

print("\n--- Keyword hits (showing matching lines) ---")
hits = 0
for i, line in enumerate(lines):
    low_line = line.lower()
    if any(k in low_line for k in keywords):
        hits += 1
        print(f"{i:4d}: {line.strip()[:200]}")
        if hits >= 80:   # limit output so terminal doesn't explode
            print("... (stopped after 80 hits)")
            break

if hits == 0:
    print("No keyword hits found.")
# --- Find the first numeric data row after the frequency header ---
import re

def is_numeric_row(line: str) -> bool:
    # BearinX rows look like: "0.0000e+00; 1.23e-04; ...;"
    parts = [p.strip() for p in line.split(";") if p.strip() != ""]
    if len(parts) < 5:
        return False
    # Check first few parts are numbers
    for p in parts[:5]:
        if not re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", p):
            return False
    return True

start_search = idx_freq if idx_freq is not None else 0

first_data_idx = None
for i in range(start_search, len(lines)):
    if is_numeric_row(lines[i]):
        first_data_idx = i
        break

print("\n--- First numeric data row ---")
print("first_data_idx:", first_data_idx)

if first_data_idx is not None:
    print("Line content (first 200 chars):")
    print(lines[first_data_idx].strip()[:200])
    parts = [p.strip() for p in lines[first_data_idx].split(";") if p.strip() != ""]
    print("Number of columns in this row:", len(parts))

