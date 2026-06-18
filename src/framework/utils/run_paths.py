from __future__ import annotations

from pathlib import Path


def get_latest_run_dir(project_root: Path) -> Path:
    latest_file = project_root / "runs" / "latest_run.txt"
    if not latest_file.exists():
        raise FileNotFoundError(
            f"Could not find {latest_file}. Run the pipeline first to create it."
        )

    run_dir_str = latest_file.read_text(encoding="utf-8").strip()
    if not run_dir_str:
        raise ValueError(f"{latest_file} is empty.")

    run_dir = (project_root / run_dir_str) if not Path(run_dir_str).is_absolute() else Path(run_dir_str)
    if not run_dir.exists():
        raise FileNotFoundError(f"Latest run dir does not exist: {run_dir}")

    return run_dir


