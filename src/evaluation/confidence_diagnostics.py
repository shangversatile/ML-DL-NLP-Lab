"""Confidence diagnostics for multiclass classifiers."""

from numbers import Integral

import numpy as np

from src.evaluation.multiclass_metrics import prediction_confidence


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
        raise ValueError("probability rows must sum to one.")


def _validate_integer_labels(
    y_true: np.ndarray,
    num_classes: int,
) -> None:
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")
    if y_true.shape[0] == 0:
        raise ValueError("y_true must not be empty.")
    if np.issubdtype(y_true.dtype, np.bool_) or not np.issubdtype(
        y_true.dtype,
        np.integer,
    ):
        raise ValueError("y_true must contain integer labels.")
    if np.any(y_true < 0) or np.any(y_true >= num_classes):
        raise ValueError("y_true labels must satisfy 0 <= label < num_classes.")


def _validate_n_bins(
    n_bins: int,
) -> None:
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, Integral):
        raise TypeError("n_bins must be an integer.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")


def _validate_inputs(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    _validate_probability_matrix(probabilities)
    _validate_integer_labels(y_true, probabilities.shape[1])
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have matching sample counts.")


def confidence_bin_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """
    Summarize accuracy and confidence within equally spaced confidence bins.
    """
    _validate_n_bins(n_bins)
    _validate_inputs(y_true, probabilities)

    confidences = prediction_confidence(probabilities)
    predictions = np.argmax(probabilities, axis=1)
    correctness = predictions == y_true
    summary = []

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == 0:
            bin_mask = (confidences >= lower) & (confidences <= upper)
        else:
            bin_mask = (confidences > lower) & (confidences <= upper)

        count = int(np.sum(bin_mask))
        if count == 0:
            accuracy = float(np.nan)
            mean_confidence = float(np.nan)
            confidence_gap = float(np.nan)
        else:
            accuracy = float(np.mean(correctness[bin_mask]))
            mean_confidence = float(np.mean(confidences[bin_mask]))
            confidence_gap = float(mean_confidence - accuracy)

        summary.append(
            {
                "bin_index": int(bin_index),
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "confidence_gap": confidence_gap,
            }
        )

    return summary


def expected_calibration_error(
    bin_summary: list[dict[str, float | int]],
    n_samples: int,
) -> float:
    """
    Compute an ECE-style diagnostic from a bin summary.
    """
    if not isinstance(bin_summary, list) or len(bin_summary) == 0:
        raise ValueError("bin_summary must be a non-empty list.")
    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, Integral):
        raise TypeError("n_samples must be an integer.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    ece = 0.0
    for bin_record in bin_summary:
        count = int(bin_record["count"])
        if count == 0:
            continue
        mean_confidence = float(bin_record["mean_confidence"])
        accuracy = float(bin_record["accuracy"])
        ece += (count / n_samples) * abs(mean_confidence - accuracy)

    return float(ece)


def summarize_confidence_behavior(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float | int]:
    """
    Summarize confidence behavior with accuracy, mean confidence, and ECE-style gap.
    """
    _validate_n_bins(n_bins)
    _validate_inputs(y_true, probabilities)

    confidences = prediction_confidence(probabilities)
    predictions = np.argmax(probabilities, axis=1)
    correctness = predictions == y_true
    n_samples = int(y_true.shape[0])
    accuracy = float(np.mean(correctness))
    mean_confidence = float(np.mean(confidences))

    correct_confidences = confidences[correctness]
    incorrect_confidences = confidences[~correctness]
    bin_summary = confidence_bin_summary(
        y_true,
        probabilities,
        n_bins=n_bins,
    )

    return {
        "n_samples": n_samples,
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "mean_confidence_correct": (
            float(np.mean(correct_confidences))
            if correct_confidences.size > 0
            else float(np.nan)
        ),
        "mean_confidence_incorrect": (
            float(np.mean(incorrect_confidences))
            if incorrect_confidences.size > 0
            else float(np.nan)
        ),
        "overconfidence_gap": float(mean_confidence - accuracy),
        "ece": expected_calibration_error(bin_summary, n_samples),
    }
