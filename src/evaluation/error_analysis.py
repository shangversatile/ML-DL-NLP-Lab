"""Error analysis utilities.

This file will be implemented in Week 4.
"""

from numbers import Integral, Real

import numpy as np

from src.evaluation.multiclass_metrics import (
    per_sample_negative_log_likelihood,
    prediction_confidence,
)


def _validate_threshold(
    threshold: float,
) -> None:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("threshold must be numeric and not boolean.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")


def _validate_top_n(
    top_n: int,
) -> None:
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, Integral):
        raise TypeError("top_n must be an integer.")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")


def _validate_prediction_vectors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a NumPy array.")
    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be one-dimensional.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")


def _mean_or_nan(
    values: np.ndarray,
) -> float:
    if values.size == 0:
        return float(np.nan)
    return float(np.mean(values))


def _example_record(
    index: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    confidence: np.ndarray,
    nll: np.ndarray,
) -> dict[str, float | int]:
    return {
        "index": int(index),
        "true_label": int(y_true[index]),
        "predicted_label": int(y_pred[index]),
        "confidence": float(confidence[index]),
        "true_class_probability": float(probabilities[index, y_true[index]]),
        "negative_log_likelihood": float(nll[index]),
        "is_error": bool(y_true[index] != y_pred[index]),
    }


def misclassification_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Return boolean mask for misclassified examples.
    """
    _validate_prediction_vectors(y_true, y_pred)
    return y_true != y_pred


def summarize_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    high_confidence_threshold: float = 0.90,
) -> dict[str, float | int]:
    """
    Summarize aggregate error and confidence behavior.
    """
    _validate_threshold(high_confidence_threshold)

    error_mask = misclassification_mask(y_true, y_pred)
    confidence = prediction_confidence(probabilities)
    nll = per_sample_negative_log_likelihood(y_true, probabilities)

    if confidence.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities must have one row per label.")

    correct_mask = ~error_mask
    high_confidence_error_mask = error_mask & (
        confidence >= high_confidence_threshold
    )

    n_samples = int(y_true.shape[0])
    n_errors = int(np.sum(error_mask))
    n_high_confidence_errors = int(np.sum(high_confidence_error_mask))

    return {
        "n_samples": n_samples,
        "n_errors": n_errors,
        "error_rate": float(n_errors / n_samples),
        "mean_confidence": float(np.mean(confidence)),
        "mean_confidence_correct": _mean_or_nan(confidence[correct_mask]),
        "mean_confidence_incorrect": _mean_or_nan(confidence[error_mask]),
        "n_high_confidence_errors": n_high_confidence_errors,
        "high_confidence_error_rate": float(n_high_confidence_errors / n_samples),
        "high_confidence_error_rate_among_errors": (
            float(n_high_confidence_errors / n_errors)
            if n_errors > 0
            else float(np.nan)
        ),
        "mean_nll": float(np.mean(nll)),
        "mean_nll_correct": _mean_or_nan(nll[correct_mask]),
        "mean_nll_incorrect": _mean_or_nan(nll[error_mask]),
    }


def select_top_loss_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    top_n: int,
) -> list[dict[str, float | int]]:
    """
    Select examples with the largest per-sample NLL.
    """
    _validate_top_n(top_n)

    error_mask = misclassification_mask(y_true, y_pred)
    confidence = prediction_confidence(probabilities)
    nll = per_sample_negative_log_likelihood(y_true, probabilities)

    if confidence.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities must have one row per label.")

    _ = error_mask
    sorted_indices = np.argsort(-nll)
    selected_indices = sorted_indices[: min(top_n, len(sorted_indices))]

    return [
        _example_record(
            int(index),
            y_true,
            y_pred,
            probabilities,
            confidence,
            nll,
        )
        for index in selected_indices
    ]


def select_high_confidence_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    top_n: int,
) -> list[dict[str, float | int]]:
    """
    Select high-confidence misclassified examples sorted by confidence.
    """
    _validate_threshold(threshold)
    _validate_top_n(top_n)

    error_mask = misclassification_mask(y_true, y_pred)
    confidence = prediction_confidence(probabilities)
    nll = per_sample_negative_log_likelihood(y_true, probabilities)

    if confidence.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities must have one row per label.")

    candidate_indices = np.flatnonzero(error_mask & (confidence >= threshold))
    sorted_candidate_indices = candidate_indices[
        np.argsort(-confidence[candidate_indices])
    ]
    selected_indices = sorted_candidate_indices[
        : min(top_n, len(sorted_candidate_indices))
    ]

    return [
        _example_record(
            int(index),
            y_true,
            y_pred,
            probabilities,
            confidence,
            nll,
        )
        for index in selected_indices
    ]
