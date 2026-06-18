"""Multiclass evaluation metrics implemented with NumPy."""

from numbers import Integral, Real

import numpy as np


def _validate_num_classes(
    num_classes: int,
) -> None:
    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(
        num_classes,
        Integral,
    ):
        raise TypeError("num_classes must be an integer.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")


def _validate_integer_labels(
    name: str,
    labels: np.ndarray,
    num_classes: int,
) -> None:
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(labels) == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.issubdtype(labels.dtype, np.bool_) or not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise ValueError(f"{name} must contain integer labels.")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"{name} labels must satisfy 0 <= label < num_classes.")


def _validate_probability_matrix(
    probabilities: np.ndarray,
) -> None:
    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities must be a NumPy array.")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional.")
    if probabilities.shape[0] == 0:
        raise ValueError("probabilities must contain at least one sample.")
    if probabilities.shape[1] < 2:
        raise ValueError("probabilities must contain at least two classes.")
    if np.issubdtype(probabilities.dtype, np.bool_) or not np.issubdtype(
        probabilities.dtype,
        np.number,
    ):
        raise ValueError("probabilities must contain numeric values.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative.")

    row_sums = np.sum(probabilities, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("each probability row must sum to one.")


def _validate_confusion_matrix(
    matrix: np.ndarray,
    require_positive_total: bool = False,
) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a NumPy array.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square and two-dimensional.")
    if matrix.shape[0] < 2:
        raise ValueError("matrix must contain at least two classes.")
    if np.issubdtype(matrix.dtype, np.bool_) or not np.issubdtype(
        matrix.dtype,
        np.number,
    ):
        raise ValueError("matrix must contain numeric values.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must contain only finite values.")
    if np.any(matrix < 0):
        raise ValueError("matrix values must be non-negative.")
    if require_positive_total and np.sum(matrix) <= 0:
        raise ValueError("matrix total count must be positive.")


def _validate_positive_epsilon(
    epsilon: float,
) -> None:
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be numeric and not boolean.")
    if not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute a multiclass confusion matrix.

    Rows correspond to true labels.
    Columns correspond to predicted labels.
    """
    _validate_num_classes(num_classes)
    _validate_integer_labels("y_true", y_true, num_classes)
    _validate_integer_labels("y_pred", y_pred, num_classes)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same sample count.")

    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def accuracy_from_confusion_matrix(
    matrix: np.ndarray,
) -> float:
    _validate_confusion_matrix(matrix, require_positive_total=True)
    return float(np.trace(matrix) / np.sum(matrix))


def per_class_recall(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute per-class recall from a confusion matrix.
    """
    _validate_confusion_matrix(matrix)

    true_support = np.sum(matrix, axis=1)
    recall = np.full(matrix.shape[0], np.nan, dtype=float)
    np.divide(
        np.diag(matrix),
        true_support,
        out=recall,
        where=true_support > 0,
    )
    return recall


def per_class_precision(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute per-class precision from a confusion matrix.
    """
    _validate_confusion_matrix(matrix)

    predicted_support = np.sum(matrix, axis=0)
    precision = np.full(matrix.shape[0], np.nan, dtype=float)
    np.divide(
        np.diag(matrix),
        predicted_support,
        out=precision,
        where=predicted_support > 0,
    )
    return precision


def macro_average(
    values: np.ndarray,
) -> float:
    """
    Compute mean value while ignoring NaN entries.
    """
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a NumPy array.")
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if values.size == 0:
        raise ValueError("values must not be empty.")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("values must contain numeric values.")
    if not np.all(np.isfinite(values) | np.isnan(values)):
        raise ValueError("values must contain only finite values or NaN.")
    if np.all(np.isnan(values)):
        raise ValueError("values must contain at least one non-NaN value.")

    return float(np.nanmean(values))


def top_k_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    k: int,
) -> float:
    """
    Compute top-k accuracy from class probabilities.
    """
    _validate_probability_matrix(probabilities)

    num_classes = probabilities.shape[1]
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        raise TypeError("k must be an integer.")
    if k < 1 or k > num_classes:
        raise ValueError("k must satisfy 1 <= k <= num_classes.")

    _validate_integer_labels("y_true", y_true, num_classes)
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have the same sample count.")

    top_k_labels = np.argsort(
        -probabilities,
        axis=1,
    )[:, :k]
    matches = np.any(top_k_labels == y_true[:, np.newaxis], axis=1)
    return float(np.mean(matches))


def prediction_confidence(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Return max predicted probability for each sample.
    """
    _validate_probability_matrix(probabilities)
    return np.max(probabilities, axis=1)


def per_sample_negative_log_likelihood(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float = 1e-15,
) -> np.ndarray:
    """
    Return per-sample negative log probability of the true class.
    """
    _validate_probability_matrix(probabilities)
    _validate_integer_labels("y_true", y_true, probabilities.shape[1])
    _validate_positive_epsilon(epsilon)

    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have the same sample count.")

    clipped = np.clip(probabilities, epsilon, 1.0)
    return -np.log(clipped[np.arange(len(y_true)), y_true])
