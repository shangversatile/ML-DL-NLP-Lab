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


def _validate_binary_classification_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Validate inputs for binary classification metrics."""
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a NumPy array.")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only 0 and 1 values.")
    if not np.all((y_pred == 0) | (y_pred == 1)):
        raise ValueError("y_pred must contain only 0 and 1 values.")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute a 2x2 binary confusion matrix.

    Return format:
    [[TN, FP],
     [FN, TP]]
    """
    _validate_binary_classification_inputs(y_true, y_pred)

    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    true_positive = np.sum((y_true == 1) & (y_pred == 1))

    return np.array(
        [[true_negative, false_positive], [false_negative, true_positive]]
    )


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary classification accuracy."""
    _validate_binary_classification_inputs(y_true, y_pred)

    accuracy = np.mean(y_true == y_pred)
    return float(accuracy)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute binary precision: TP / (TP + FP).

    If TP + FP == 0, return 0.0.
    """
    matrix = confusion_matrix(y_true, y_pred)
    false_positive = matrix[0, 1]
    true_positive = matrix[1, 1]
    denominator = true_positive + false_positive

    if denominator == 0:
        return 0.0
    return float(true_positive / denominator)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute binary recall: TP / (TP + FN).

    If TP + FN == 0, return 0.0.
    """
    matrix = confusion_matrix(y_true, y_pred)
    false_negative = matrix[1, 0]
    true_positive = matrix[1, 1]
    denominator = true_positive + false_negative

    if denominator == 0:
        return 0.0
    return float(true_positive / denominator)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute binary F1 score.

    If precision + recall == 0, return 0.0.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    denominator = precision + recall

    if denominator == 0:
        return 0.0
    return float(2 * precision * recall / denominator)
