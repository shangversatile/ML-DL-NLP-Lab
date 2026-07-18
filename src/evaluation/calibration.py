"""Calibration metrics and reliability-diagram utilities for multiclass classifiers."""

from __future__ import annotations

import numpy as np


def _validate_n_bins(n_bins: int) -> int:
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins,
        (int, np.integer),
    ):
        raise TypeError("n_bins must be an integer.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    return int(n_bins)


def _validate_positive_epsilon(epsilon: float) -> float:
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(
        epsilon,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("epsilon must be numeric and not boolean.")
    epsilon_float = float(epsilon)
    if not np.isfinite(epsilon_float):
        raise ValueError("epsilon must be finite.")
    if epsilon_float <= 0.0:
        raise ValueError("epsilon must be positive.")
    return epsilon_float


def validate_probabilities_and_labels(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate multiclass probability predictions and integer labels.
    """
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

    validated_probabilities = probabilities.astype(float, copy=True)
    if not np.all(np.isfinite(validated_probabilities)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(validated_probabilities < 0.0) or np.any(
        validated_probabilities > 1.0,
    ):
        raise ValueError("probabilities must contain values in [0, 1].")

    row_sums = np.sum(validated_probabilities, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("probability rows must sum to one.")

    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a NumPy array.")
    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")
    if y_true.shape[0] == 0:
        raise ValueError("y_true must not be empty.")
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("probabilities and y_true must have matching sample counts.")
    if np.issubdtype(y_true.dtype, np.bool_) or not np.issubdtype(
        y_true.dtype,
        np.number,
    ):
        raise ValueError("y_true must contain integer-like labels.")

    y_float = y_true.astype(float, copy=False)
    if not np.all(np.isfinite(y_float)):
        raise ValueError("y_true must contain only finite labels.")
    if not np.all(y_float == np.floor(y_float)):
        raise ValueError("y_true must contain integer-like labels.")

    validated_labels = y_float.astype(int, copy=True)
    n_classes = validated_probabilities.shape[1]
    if np.any(validated_labels < 0) or np.any(validated_labels >= n_classes):
        raise ValueError("y_true labels must satisfy 0 <= label < n_classes.")

    return validated_probabilities, validated_labels


def prediction_confidence_and_correctness(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return predicted labels, confidences, and correctness indicators.
    """
    validated_probabilities, validated_labels = validate_probabilities_and_labels(
        probabilities,
        y_true,
    )
    predictions = np.argmax(validated_probabilities, axis=1)
    confidences = np.max(validated_probabilities, axis=1)
    correctness = predictions == validated_labels
    return predictions, confidences, correctness


def calibration_bin_summary(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """
    Compute confidence-bin statistics for reliability diagrams.

    The returned gap is signed: average confidence minus empirical accuracy.
    Positive gaps indicate overconfidence within the bin.
    """
    validated_n_bins = _validate_n_bins(n_bins)
    _, confidences, correctness = prediction_confidence_and_correctness(
        probabilities,
        y_true,
    )

    summary = []
    for bin_index in range(validated_n_bins):
        lower = bin_index / validated_n_bins
        upper = (bin_index + 1) / validated_n_bins
        if bin_index == validated_n_bins - 1:
            bin_mask = (confidences >= lower) & (confidences <= upper)
        else:
            bin_mask = (confidences >= lower) & (confidences < upper)

        count = int(np.sum(bin_mask))
        if count == 0:
            accuracy = float(np.nan)
            confidence = float(np.nan)
            gap = float(np.nan)
        else:
            accuracy = float(np.mean(correctness[bin_mask]))
            confidence = float(np.mean(confidences[bin_mask]))
            gap = float(confidence - accuracy)

        summary.append(
            {
                "bin_index": int(bin_index),
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": gap,
            }
        )

    return summary


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute top-label expected calibration error.
    """
    validated_probabilities, validated_labels = validate_probabilities_and_labels(
        probabilities,
        y_true,
    )
    summary = calibration_bin_summary(
        validated_probabilities,
        validated_labels,
        n_bins=n_bins,
    )
    n_samples = validated_probabilities.shape[0]

    ece = 0.0
    for record in summary:
        count = int(record["count"])
        if count == 0:
            continue
        ece += (count / n_samples) * abs(float(record["gap"]))

    return float(ece)


def maximum_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute maximum absolute calibration gap across non-empty bins.
    """
    validated_probabilities, validated_labels = validate_probabilities_and_labels(
        probabilities,
        y_true,
    )
    summary = calibration_bin_summary(
        validated_probabilities,
        validated_labels,
        n_bins=n_bins,
    )

    gaps = [
        abs(float(record["gap"]))
        for record in summary
        if int(record["count"]) > 0
    ]
    return float(max(gaps))


def multiclass_brier_score(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Compute multiclass Brier score using one-hot labels.
    """
    validated_probabilities, validated_labels = validate_probabilities_and_labels(
        probabilities,
        y_true,
    )
    one_hot = np.zeros_like(validated_probabilities, dtype=float)
    one_hot[np.arange(validated_labels.shape[0]), validated_labels] = 1.0
    squared_l2_errors = np.sum((validated_probabilities - one_hot) ** 2, axis=1)
    return float(np.mean(squared_l2_errors))


def negative_log_likelihood(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    """
    Compute multiclass negative log likelihood.
    """
    epsilon_float = _validate_positive_epsilon(epsilon)
    validated_probabilities, validated_labels = validate_probabilities_and_labels(
        probabilities,
        y_true,
    )
    clipped = np.clip(validated_probabilities, epsilon_float, 1.0)
    true_class_probabilities = clipped[
        np.arange(validated_labels.shape[0]),
        validated_labels,
    ]
    return float(np.mean(-np.log(true_class_probabilities)))


def calibration_summary(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float | int]:
    """
    Compute accuracy, mean confidence, ECE, MCE, Brier score, and NLL.
    """
    validated_probabilities, validated_labels = validate_probabilities_and_labels(
        probabilities,
        y_true,
    )
    _, confidences, correctness = prediction_confidence_and_correctness(
        validated_probabilities,
        validated_labels,
    )

    n_samples = int(validated_labels.shape[0])
    accuracy = float(np.mean(correctness))
    mean_confidence = float(np.mean(confidences))

    correct_confidences = confidences[correctness]
    incorrect_confidences = confidences[~correctness]

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
        "ece": expected_calibration_error(
            validated_probabilities,
            validated_labels,
            n_bins=n_bins,
        ),
        "mce": maximum_calibration_error(
            validated_probabilities,
            validated_labels,
            n_bins=n_bins,
        ),
        "brier_score": multiclass_brier_score(
            validated_probabilities,
            validated_labels,
        ),
        "nll": negative_log_likelihood(
            validated_probabilities,
            validated_labels,
        ),
    }
