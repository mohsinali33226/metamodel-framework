import re
import pandas as pd
from pathlib import Path

base = Path("./data/outputs/l_effective/adaptive_bearinx_run_batch_80")
sampling = pd.read_csv(base / "sampling_with_ts.csv")

template_path = Path("data/raw/Bearinx_input_vg2/output_RiKuLa_00000.vg2")
template_lines = template_path.read_text(errors="ignore").splitlines()

outdir = base / "vg2_files"
outdir.mkdir(parents=True, exist_ok=True)

def fmt(v):
    return format(float(v), ".15g")

def replace_var_line(line, value, trailing_zero=False):
    if trailing_zero:
        return re.sub(r"\{[^}]*\}", f"{{ {fmt(value)} 0 }}", line)
    return re.sub(r"\{[^}]*\}", f"{{ {fmt(value)} }}", line)

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

for ln in [13454, 13479, 13504, 13529, 13554, 13579, 13604, 13629, 13654]:
    line_map[ln] = ("Dw", False)

line_map[17278] = ("Fa", True)
line_map[17279] = ("Fr", True)

for idx, row in sampling.iterrows():
    new_lines = []

    for line_no, line in enumerate(template_lines, start=1):
        if line_no in line_map:
            col, trailing_zero = line_map[line_no]
            val = int(round(row[col])) if col == "Z" else row[col]
            new_lines.append(replace_var_line(line, val, trailing_zero))
        else:
            new_lines.append(line)

    out = outdir / f"adaptive_RiKuLa_{idx:05d}.vg2"
    out.write_text("\n".join(new_lines), encoding="utf-8")

print("created vg2 files:", len(list(outdir.glob('*.vg2'))))
print("saved folder:", outdir)

