from pathlib import Path
import re

def next_run_dir(base_dir: Path, prefix: str) -> Path:
    """
    Create the next free run directory like:
    runs/csv_benchmark_v1, runs/csv_benchmark_v2, ...
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)$")

    max_v = 0
    for p in base_dir.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                max_v = max(max_v, int(m.group(1)))

    return base_dir / f"{prefix}_v{max_v + 1}"


