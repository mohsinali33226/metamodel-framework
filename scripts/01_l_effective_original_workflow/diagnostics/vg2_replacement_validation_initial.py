import re
import pandas as pd
from pathlib import Path

sampling = pd.read_csv("data/raw/sampling/sampling.csv")
template_path = Path("data/raw/Bearinx_input_vg2/output_RiKuLa_00000.vg2")
target_path = Path("data/raw/Bearinx_input_vg2/output_RiKuLa_00001.vg2")

row = sampling.iloc[1]

lines = template_path.read_text(errors="ignore").splitlines()
target_lines = target_path.read_text(errors="ignore").splitlines()

def replace_var_line(line, value, trailing_zero=False):
    if trailing_zero:
        return re.sub(r"\{[^}]*\}", f"{{ {value} 0 }}", line)
    return re.sub(r"\{[^}]*\}", f"{{ {value} }}", line)

# variable-wide replacements
var_map = {
    "IDLD_FY": ("Fr", True),
    "IDLD_FX": ("Fa", True),
    "IDLC_SPEED": ("n", False),
    "IDL_OILTEMP": ("T", True),
    "IDR_RADIAL_CLEARANCE": ("s", True),
    "IDR_NBROLLINGELEMENTS": ("Z", False),
    "IDL_MAX_CAGE_EXCENTRICITY": ("e_c_norm", False),
    "IDW_DW": ("Dw", False),
    "IDR_RA_WK_INPUT": ("Ra_w", False),
    "IDR_SVI_NEW": ("SVi", False),
    "IDR_SVA_NEW": ("SVa", False),
    "IDL_SIMULATION_TIME_FOR_NOISE_CALCULATION": ("t_s", True),
}

new_lines = []

for line_no, line in enumerate(lines, start=1):
    new_line = line

    # line-specific dSi/dSa replacements because both use IDI_DSH
    if line_no == 12856:
        new_line = replace_var_line(line, row["dSi"], trailing_zero=False)
    elif line_no == 13113:
        new_line = replace_var_line(line, row["dSa"], trailing_zero=False)
    else:
        for var, (col, trailing_zero) in var_map.items():
            if f"VARIABLE {var} " in line:
                val = int(round(row[col])) if col == "Z" else row[col]
                new_line = replace_var_line(line, val, trailing_zero=trailing_zero)
                break

    new_lines.append(new_line)

# compare only mapped lines
mismatch = []
for i, (a, b) in enumerate(zip(new_lines, target_lines), start=1):
    if i in [12856, 13113] or any(f"VARIABLE {v} " in a for v in var_map):
        if a.strip() != b.strip():
            mismatch.append((i, a.strip(), b.strip()))

print("mapped_line_mismatches =", len(mismatch))
for m in mismatch[:30]:
    print("\nLINE", m[0])
    print("generated:", m[1])
    print("target:   ", m[2])

