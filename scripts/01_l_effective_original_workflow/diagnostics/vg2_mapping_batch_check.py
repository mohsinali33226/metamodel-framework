import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

sampling = pd.read_csv("data/raw/sampling/sampling.csv")
vg2_dir = Path("data/raw/Bearinx_input_vg2")

num_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

def nums(line):
    return [float(x) for x in num_re.findall(line)]

def close(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))

def variable_name(line):
    m = re.search(r"VARIABLE\s+([A-Za-z0-9_]+)", line)
    return m.group(1) if m else "UNKNOWN"

hits = defaultdict(list)

# compare several consecutive file pairs
for idx in range(0, 20):
    f0 = vg2_dir / f"output_RiKuLa_{idx:05d}.vg2"
    f1 = vg2_dir / f"output_RiKuLa_{idx+1:05d}.vg2"

    if not f0.exists() or not f1.exists():
        continue

    row0 = sampling.iloc[idx]
    row1 = sampling.iloc[idx + 1]

    lines0 = f0.read_text(errors="ignore").splitlines()
    lines1 = f1.read_text(errors="ignore").splitlines()

    for line_no, (a, b) in enumerate(zip(lines0, lines1), start=1):
        if a == b:
            continue

        na = nums(a)
        nb = nums(b)

        for col in sampling.columns:
            v0 = row0[col]
            v1 = row1[col]

            # skip if value did not change between rows
            if close(v0, v1):
                continue

            if any(close(x, v0) for x in na) and any(close(y, v1) for y in nb):
                hits[col].append((idx, line_no, variable_name(a), a.strip(), b.strip()))

print("ROBUST VG2 MAPPING CANDIDATES")
print("=" * 100)

for col in sampling.columns:
    print("\nCOL:", col)
    items = hits.get(col, [])
    if not items:
        print("  NO HITS")
        continue

    # count variable names
    counts = {}
    for _, _, var, _, _ in items:
        counts[var] = counts.get(var, 0) + 1

    print("  variable_counts:", counts)

    # show first 5 examples
    for ex in items[:5]:
        idx, line_no, var, old, new = ex
        print(f"  pair {idx}->{idx+1}, line {line_no}, var {var}")
        print("   ", old)
        print("   ", new)

