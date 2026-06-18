import re
import pandas as pd
from pathlib import Path

sampling = pd.read_csv("data/raw/sampling/sampling.csv")

row0 = sampling.iloc[0]
row1 = sampling.iloc[1]

f0 = Path("data/raw/Bearinx_input_vg2/output_RiKuLa_00000.vg2")
f1 = Path("data/raw/Bearinx_input_vg2/output_RiKuLa_00001.vg2")

lines0 = f0.read_text(errors="ignore").splitlines()
lines1 = f1.read_text(errors="ignore").splitlines()

num_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

def nums(line):
    return [float(x) for x in num_re.findall(line)]

def close(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))

print("Sampling row 0:")
print(row0.to_string())
print("\nSampling row 1:")
print(row1.to_string())

print("\nPossible changed VG2 lines matching sampling row0 -> row1 values:")
print("=" * 90)

for i, (a, b) in enumerate(zip(lines0, lines1), start=1):
    if a == b:
        continue

    na = nums(a)
    nb = nums(b)

    matches = []
    for col in sampling.columns:
        v0 = row0[col]
        v1 = row1[col]

        if any(close(x, v0) for x in na) and any(close(y, v1) for y in nb):
            matches.append(col)

    if matches:
        print("\nLINE", i, "MATCH_COLS:", matches)
        print("00000:", a.strip())
        print("00001:", b.strip())

