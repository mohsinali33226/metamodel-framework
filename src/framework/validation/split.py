import numpy as np
from sklearn.model_selection import KFold



def train_test_split(X, y, test_size: float = 0.2, seed: int = 42, shuffle: bool = True):
    """
    Simple train/test split.

    Returns: X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n = X.shape[0]
    if n != y.shape[0]:
        raise ValueError(f"X and y must have same number of rows. Got {n} and {y.shape[0]}")

    idx = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_test = int(round(n * test_size))
    n_test = max(1, min(n - 1, n_test))  # keep at least 1 train and 1 test

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
def kfold_splits(x, y, n_splits: int = 5, seed: int = 42, shuffle: bool = True):
    from sklearn.model_selection import KFold

    if len(x) != len(y):
        raise ValueError("x and y must have the same number of rows.")

    x = np.asarray(x)
    y = np.asarray(y)

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for train_idx, test_idx in kf.split(x):
        yield train_idx, test_idx



def group_kfold_splits(X, y, groups, n_splits: int = 5, seed: int = 42):
    """
    Yield (train_idx, test_idx) for GroupKFold.
    Note: GroupKFold does not shuffle (seed kept for future extension).
    """
    from sklearn.model_selection import GroupKFold

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if X.shape[0] != y.shape[0] or X.shape[0] != groups.shape[0]:
        raise ValueError("X, y, and groups must have the same number of rows.")

    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        yield train_idx, test_idx


def split_indices(
    method: str,
    X,
    y,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    shuffle: bool = True,
    groups=None,
    n_splits: int = 5,
):
    """
    Generator of (train_idx, test_idx) based on method.

    # method options:
# - "train_test_split": yields exactly one split
# - "kfold": yields n_splits splits
# - "group_kfold": yields n_splits splits (requires groups)

    """
    method = str(method).strip().lower()
    X = np.asarray(X)
    y = np.asarray(y)

    n = X.shape[0]
    if n != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    if method == "train_test_split":
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

        n_test = int(round(n * test_size))
        n_test = max(1, min(n - 1, n_test))

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        yield train_idx, test_idx
        return
    if method == "kfold":
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        folds = np.array_split(idx, n_splits)
        for k in range(n_splits):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != k])
            yield train_idx, test_idx
        return


    if method == "group_kfold":
        if groups is None:
            raise ValueError("groups must be provided for group_kfold.")
        yield from group_kfold_splits(X, y, groups=groups, n_splits=n_splits, seed=seed)
        return

    raise ValueError(f"Unknown split method: {method}")


