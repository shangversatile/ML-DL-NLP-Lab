"""Utilities for loading the handwritten digits dataset."""

import numpy as np
from sklearn.datasets import load_digits


def load_digits_dataset(
    scale_pixels: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the scikit-learn handwritten digits dataset.

    Returns:
        X: flattened feature matrix of shape (n_samples, 64)
        y: integer labels of shape (n_samples,)
        images: image tensor of shape (n_samples, 8, 8)
    """
    digits = load_digits()

    X = digits.data.astype(float)
    y = digits.target.astype(int)
    images = digits.images.astype(float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if images.ndim != 3:
        raise ValueError("images must be a 3D array.")
    if X.shape[0] != y.shape[0] or X.shape[0] != images.shape[0]:
        raise ValueError("X, y, and images must have matching sample counts.")
    if X.shape[1] != 64:
        raise ValueError("X must contain 64 flattened pixel features.")
    if images.shape[1:] != (8, 8):
        raise ValueError("images must have shape (n_samples, 8, 8).")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must contain integer values.")
    if np.any(y < 0) or np.any(y > 9):
        raise ValueError("labels must be integers in the range [0, 9].")

    if scale_pixels:
        X = X / 16.0
        images = images / 16.0

    return X, y, images
