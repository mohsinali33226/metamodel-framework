from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class CSVLoadConfig:
    path: Path
    sep: Optional[str] = None          # None -> auto-detect "," vs ";"
    encoding: str = "utf-8"
    decimal: Optional[str] = None      # None -> auto-detect "." vs ","
    low_memory: bool = False
    id_col: Optional[str] = None



def _detect_sep_and_decimal(sample_text: str) -> tuple[str, str]:
    # very small, practical heuristic for our two known formats:
    # - sampling.csv tends to be comma-separated
    # - results.csv in Fabian style uses semicolon and decimal "."
    comma_count = sample_text.count(",")
    semi_count = sample_text.count(";")

    sep = ";" if semi_count > comma_count else ","

    # decimal detection: if we use ";" as separator, decimals are almost always "."
    # if we use "," as separator, decimals could be "." (common) or "," (rare in our case)
    decimal = "." if sep == ";" else "."
    return sep, decimal


def load_csv(cfg: CSVLoadConfig) -> pd.DataFrame:
    path = Path(cfg.path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # read a small chunk to detect format if not provided
    from itertools import islice

    # read a small chunk to detect format if not provided
    if cfg.sep is None or cfg.decimal is None:
        with open(path, "r", encoding=cfg.encoding, errors="ignore") as f:
            sample_text = "".join(islice(f, 20))

        sep_detected, dec_detected = _detect_sep_and_decimal(sample_text)
        sep = cfg.sep or sep_detected
        decimal = cfg.decimal or dec_detected
    else:
        sep = cfg.sep
        decimal = cfg.decimal


    df = pd.read_csv(
        path,
        sep=sep,
        encoding=cfg.encoding,
        encoding_errors="ignore",
        decimal=decimal,
        low_memory=cfg.low_memory,
    )
    if cfg.id_col is not None and cfg.id_col not in df.columns:
        raise ValueError(f"id_col '{cfg.id_col}' not found in CSV columns.")

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # minimal cleaning aligned with student scripts:
    # - keep column names as-is (important for compatibility)
    # - drop fully empty columns
    df = df.copy()
    df = df.dropna(axis=1, how="all")

    # strip whitespace from string column names
    df.columns = [c.strip() for c in df.columns]

    return df


