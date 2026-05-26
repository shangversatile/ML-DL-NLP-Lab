"""Evaluation metrics."""

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error between true and predicted values."""
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a NumPy array.")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse)
