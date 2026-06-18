from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.15
    random_state: int = 42
    shuffle: bool = True


def split_train_test(
    df: pd.DataFrame,
    cfg: SplitConfig = SplitConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Moritz-compatible style split:
    - 85% train/valid
    - 15% final test
    - shuffle=True, random_state=42
    """
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=cfg.shuffle,
    )
    # Keep original index out of the way (clean, stable downstream)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def grouped_kfold_splits(
    df: pd.DataFrame,
    group_col: str = "block_id",
    n_splits: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    New-bearing/generalization style:
    - uses GroupKFold based on a group column (e.g., block_id)
    - NOTE: GroupKFold does not shuffle; this matches sklearn behavior.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    groups = df[group_col].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    X_index = np.arange(len(df))
    for train_idx, val_idx in gkf.split(X_index, y=None, groups=groups):
        yield train_idx, val_idx


def decide_cv_strategy(
    df: pd.DataFrame,
    group_col: str = "block_id",
) -> str:
    """
    Helper to keep pipeline logic simple.
    Returns:
        "groupkfold" if group_col exists, otherwise "kfold"
    """
    return "groupkfold" if group_col in df.columns else "kfold"
def get_groups(df, group_col: str | None):
    """
    Return groups array for GroupKFold, or None if group_col is not provided.
    """
    if group_col is None:
        return None

    group_col = str(group_col).strip()
    if not group_col:
        return None

    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in dataframe columns.")

    return df[group_col].values



