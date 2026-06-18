"""Data preprocessing utilities."""

from collections.abc import Iterator
from numbers import Real

import numpy as np


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into train and validation sets.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    n_samples = X.shape[0]
    n_val = int(n_samples * val_ratio)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split arrays into stratified train, validation, and test sets.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a NumPy array")
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    for name, ratio in (
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if isinstance(ratio, (bool, np.bool_)):
            raise TypeError(f"{name} must be numeric and not boolean")
        if not isinstance(ratio, Real):
            raise TypeError(f"{name} must be numeric")

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0.0 and 1.0")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0.0 and 1.0")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.0")

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("y must contain at least two classes")

    rng = np.random.default_rng(seed)
    train_indices_by_class = []
    val_indices_by_class = []
    test_indices_by_class = []

    for class_label in classes:
        class_indices = np.flatnonzero(y == class_label)
        rng.shuffle(class_indices)

        n_class = len(class_indices)
        n_val = int(round(n_class * val_ratio))
        n_test = int(round(n_class * test_ratio))
        n_train = n_class - n_val - n_test

        if n_val < 1 or n_test < 1 or n_train < 1:
            raise ValueError(
                "every class must have at least one train, validation, "
                "and test sample"
            )

        val_indices_by_class.append(class_indices[:n_val])
        test_indices_by_class.append(class_indices[n_val : n_val + n_test])
        train_indices_by_class.append(class_indices[n_val + n_test :])

    train_indices = np.concatenate(train_indices_by_class)
    val_indices = np.concatenate(val_indices_by_class)
    test_indices = np.concatenate(test_indices_by_class)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices].copy(),
        X[val_indices].copy(),
        X[test_indices].copy(),
        y[train_indices].copy(),
        y[val_indices].copy(),
        y[test_indices].copy(),
    )


def iterate_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield mini-batches of X and y.

    The final batch may contain fewer than batch_size samples.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a NumPy array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError("batch_size must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        batch_indices = indices[start : start + batch_size]
        yield X[batch_indices], y[batch_indices]


def flip_binary_labels(
    y: np.ndarray,
    flip_rate: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Return a copy of binary labels with a deterministic subset flipped.

    A flipped label changes:
    0 -> 1
    1 -> 0
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a NumPy array")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1")
    if isinstance(flip_rate, (bool, np.bool_)):
        raise TypeError("flip_rate must be numeric and not boolean")
    if not isinstance(flip_rate, Real):
        raise TypeError("flip_rate must be numeric")
    if flip_rate < 0.0 or flip_rate > 1.0:
        raise ValueError("flip_rate must be between 0.0 and 1.0")

    corrupted_y = y.copy()
    n_flips = int(round(len(y) * flip_rate))

    if n_flips == 0:
        return corrupted_y

    rng = np.random.default_rng(seed)
    flip_indices = rng.choice(
        len(y),
        size=n_flips,
        replace=False,
    )
    corrupted_y[flip_indices] = 1 - corrupted_y[flip_indices]

    return corrupted_y


def standardize_features(
    X_train: np.ndarray,
    X_val: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """
    Standardize features using training-set mean and std.

    Return:
    - X_train_scaled
    - X_val_scaled or None
    - mean
    - std
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = (X_val - mean) / std

    return X_train_scaled, X_val_scaled, mean, std
