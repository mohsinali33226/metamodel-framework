from __future__ import annotations

from pathlib import Path
import yaml


def load_output_paths(path: str | Path = "outputs_paths.yaml") -> dict:
    """
    Load central output paths YAML.
    Returns a nested dict, for example:
    cfg["l_effective"]["x_test"]
    cfg["l_effective"]["models_dir"]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing outputs paths file: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("outputs_paths.yaml did not parse to a dict.")

    return cfg

