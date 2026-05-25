"""Data preprocessing utilities."""

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
