import re
import pandas as pd
from pathlib import Path

sampling = pd.read_csv("data/raw/sampling/sampling.csv")
template_path = Path("data/raw/Bearinx_input_vg2/output_RiKuLa_00000.vg2")
target_path = Path("data/raw/Bearinx_input_vg2/output_RiKuLa_00001.vg2")

row = sampling.iloc[1]

lines = template_path.read_text(errors="ignore").splitlines()
target_lines = target_path.read_text(errors="ignore").splitlines()

def fmt(v):
    return format(float(v), ".15g")

def replace_var_line(line, value, trailing_zero=False):
    if trailing_zero:
        return re.sub(r"\{[^}]*\}", f"{{ {fmt(value)} 0 }}", line)
    return re.sub(r"\{[^}]*\}", f"{{ {fmt(value)} }}", line)

# specific mapped VG2 lines, 1-based line numbers
line_map = {
    632: ("t_s", True),
    635: ("e_c_norm", False),
    655: ("T", True),
    12856: ("dSi", False),
    12958: ("n", False),
    12982: ("Fa", True),
    12983: ("Fr", True),
    13113: ("dSa", False),
    13260: ("Fa", True),
    13261: ("Fr", True),
    13350: ("s", True),
    13359: ("Z", False),
    13392: ("Ra_w", False),
    13431: ("SVi", False),
    13432: ("SVa", False),
}

# all rolling elements use Dw
for ln in [13454, 13479, 13504, 13529, 13554, 13579, 13604, 13629, 13654]:
    line_map[ln] = ("Dw", False)

# final repeated load lines
line_map[17278] = ("Fa", True)
line_map[17279] = ("Fr", True)

new_lines = []

for line_no, line in enumerate(lines, start=1):
    if line_no in line_map:
        col, trailing_zero = line_map[line_no]
        val = int(round(row[col])) if col == "Z" else row[col]
        new_lines.append(replace_var_line(line, val, trailing_zero))
    else:
        new_lines.append(line)

# numeric-tolerant validation on mapped lines only
num_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

def nums(s):
    return [float(x) for x in num_re.findall(s)]

bad = []

for ln in sorted(line_map):
    a = new_lines[ln - 1]
    b = target_lines[ln - 1]
    na = nums(a)
    nb = nums(b)

    if len(na) != len(nb):
        bad.append((ln, a.strip(), b.strip()))
        continue

    ok = all(abs(x - y) <= 1e-8 * max(1.0, abs(y)) for x, y in zip(na, nb))

    if not ok:
        bad.append((ln, a.strip(), b.strip()))

print("numeric_mapped_line_mismatches =", len(bad))

for ln, a, b in bad:
    print("\nLINE", ln)
    print("generated:", a)
    print("target:   ", b)

