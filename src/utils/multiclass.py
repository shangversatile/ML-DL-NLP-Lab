"""Multiclass probability and loss utilities implemented with NumPy."""

from numbers import Integral, Real

import numpy as np


def one_hot_encode(
    y: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Convert integer class labels into a one-hot matrix.
    """
    _validate_label_vector(y)
    _validate_num_classes(num_classes)

    num_classes = int(num_classes)
    _validate_label_range(y, num_classes)

    encoded = np.zeros((len(y), num_classes), dtype=float)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def stable_softmax(
    logits: np.ndarray,
) -> np.ndarray:
    """
    Compute row-wise softmax probabilities using a numerically stable shift.
    """
    if not isinstance(logits, np.ndarray):
        raise TypeError("logits must be a NumPy array.")
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array.")
    if logits.shape[0] == 0:
        raise ValueError("logits must contain at least one sample.")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one class.")
    if np.issubdtype(logits.dtype, np.bool_) or not np.issubdtype(
        logits.dtype,
        np.number,
    ):
        raise ValueError("logits values must be numeric.")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits values must be finite.")

    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )
    exp_shifted = np.exp(shifted_logits)
    return exp_shifted / np.sum(
        exp_shifted,
        axis=1,
        keepdims=True,
    )


def multiclass_cross_entropy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    """
    Compute mean multiclass cross entropy from integer labels and probabilities.
    """
    _validate_label_vector(y_true)
    _validate_probability_matrix(probabilities)
    _validate_positive_real("epsilon", epsilon)

    if len(y_true) != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have the same sample count.")

    _validate_label_range(y_true, probabilities.shape[1])

    clipped = np.clip(
        probabilities,
        epsilon,
        1.0,
    )
    correct_class_probabilities = clipped[
        np.arange(len(y_true)),
        y_true,
    ]
    loss = -np.mean(np.log(correct_class_probabilities))
    return float(loss)


def multiclass_cross_entropy_from_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
) -> float:
    """
    Compute mean multiclass cross entropy directly from logits.
    """
    probabilities = stable_softmax(logits)
    return multiclass_cross_entropy(y_true, probabilities)


def softmax_cross_entropy_gradient(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Compute dL/dZ for softmax followed by mean cross entropy.
    """
    _validate_label_vector(y_true)
    _validate_probability_matrix(probabilities)

    if len(y_true) != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have the same sample count.")

    one_hot_labels = one_hot_encode(y_true, probabilities.shape[1])
    return (probabilities - one_hot_labels) / len(y_true)


def _validate_label_vector(y: np.ndarray) -> None:
    """Validate a one-dimensional integer label vector."""
    if not isinstance(y, np.ndarray):
        raise TypeError("labels must be a NumPy array.")
    if y.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if len(y) == 0:
        raise ValueError("labels must contain at least one value.")
    if np.issubdtype(y.dtype, np.bool_) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must contain integer values.")


def _validate_label_range(y: np.ndarray, num_classes: int) -> None:
    """Validate that labels are in the half-open range [0, num_classes)."""
    if np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError("labels must satisfy 0 <= label < num_classes.")


def _validate_num_classes(num_classes: int) -> None:
    """Validate a positive integer class count."""
    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(num_classes, Integral):
        raise TypeError("num_classes must be a positive integer.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")


def _validate_probability_matrix(probabilities: np.ndarray) -> None:
    """Validate a row-wise categorical probability matrix."""
    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities must be a NumPy array.")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    if probabilities.shape[0] == 0:
        raise ValueError("probabilities must contain at least one sample.")
    if probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one class.")
    if np.issubdtype(probabilities.dtype, np.bool_) or not np.issubdtype(
        probabilities.dtype,
        np.number,
    ):
        raise ValueError("probability values must be numeric.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probability values must be finite.")
    if np.any(probabilities < 0.0):
        raise ValueError("probability values must be non-negative.")

    row_sums = np.sum(probabilities, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("each probability row must sum to one.")


def _validate_positive_real(name: str, value: float) -> None:
    """Validate a finite positive real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive real number.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
