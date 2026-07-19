"""Selective prediction and abstention diagnostics."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from src.evaluation.calibration import validate_probabilities_and_labels


def validate_confidence_threshold(threshold: float) -> float:
    """
    Validate a confidence threshold in [0, 1].
    """
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real scalar and not boolean.")

    threshold_float = float(threshold)
    if not np.isfinite(threshold_float):
        raise ValueError("threshold must be finite.")
    if threshold_float < 0.0 or threshold_float > 1.0:
        raise ValueError("threshold must be in [0, 1].")

    return threshold_float


def _validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
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

    return validated_probabilities


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size > 0 else float(np.nan)


def _validate_unit_interval_scalar(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar and not boolean.")

    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")
    if value_float < 0.0 or value_float > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")

    return value_float


def top_label_predictions(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return predictions, confidences, and correctness indicators.
    """
    validated_probabilities, validated_labels = validate_probabilities_and_labels(
        probabilities,
        y_true,
    )
    predictions = np.argmax(validated_probabilities, axis=1)
    confidences = np.max(validated_probabilities, axis=1)
    correctness = predictions == validated_labels
    return predictions, confidences, correctness


def abstention_mask_from_confidence(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Return boolean answered mask where max probability >= threshold.
    """
    validated_probabilities = _validate_probability_matrix(probabilities)
    threshold_float = validate_confidence_threshold(threshold)
    confidences = np.max(validated_probabilities, axis=1)
    return confidences >= threshold_float


def selective_prediction_metrics(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """
    Evaluate selective prediction under a confidence threshold.
    """
    _, confidences, correctness = top_label_predictions(probabilities, y_true)
    threshold_float = validate_confidence_threshold(threshold)

    answered_mask = confidences >= threshold_float
    abstained_mask = ~answered_mask
    error_mask = ~correctness

    n_samples = int(correctness.shape[0])
    answered_count = int(np.sum(answered_mask))
    abstained_count = int(np.sum(abstained_mask))
    errors_total = int(np.sum(error_mask))
    errors_answered = int(np.sum(error_mask & answered_mask))
    errors_abstained = int(np.sum(error_mask & abstained_mask))

    if answered_count == 0:
        selective_accuracy = float(np.nan)
        selective_error_rate = float(np.nan)
    else:
        selective_accuracy = float(np.mean(correctness[answered_mask]))
        selective_error_rate = float(1.0 - selective_accuracy)

    return {
        "n_samples": n_samples,
        "threshold": threshold_float,
        "answered_count": answered_count,
        "abstained_count": abstained_count,
        "coverage": float(answered_count / n_samples),
        "abstention_rate": float(abstained_count / n_samples),
        "original_accuracy": float(np.mean(correctness)),
        "selective_accuracy": selective_accuracy,
        "selective_error_rate": selective_error_rate,
        "errors_total": errors_total,
        "errors_answered": errors_answered,
        "errors_abstained": errors_abstained,
        "error_abstention_rate": (
            float(errors_abstained / errors_total)
            if errors_total > 0
            else float(np.nan)
        ),
        "mean_answered_confidence": _mean_or_nan(confidences[answered_mask]),
        "mean_abstained_confidence": _mean_or_nan(confidences[abstained_mask]),
    }


def threshold_sweep(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray | list[float] | None = None,
) -> list[dict[str, float | int]]:
    """
    Evaluate selective prediction over a sequence of confidence thresholds.
    """
    if thresholds is None:
        threshold_values = np.linspace(0.0, 1.0, 21)
    else:
        threshold_values = np.asarray(thresholds, dtype=object)
        if threshold_values.ndim != 1:
            raise ValueError("thresholds must be one-dimensional.")
        if threshold_values.size == 0:
            raise ValueError("thresholds must not be empty.")

    validated_thresholds = [
        validate_confidence_threshold(threshold)
        for threshold in threshold_values
    ]

    return [
        selective_prediction_metrics(probabilities, y_true, threshold)
        for threshold in validated_thresholds
    ]


def choose_threshold_for_target_selective_accuracy(
    sweep_results: list[dict[str, float | int]],
    target_accuracy: float,
    min_coverage: float = 0.0,
) -> dict[str, float | int] | None:
    """
    Choose the lowest threshold that reaches target selective accuracy
    while satisfying minimum coverage.
    """
    target_accuracy_float = _validate_unit_interval_scalar(
        "target_accuracy",
        target_accuracy,
    )
    min_coverage_float = _validate_unit_interval_scalar("min_coverage", min_coverage)

    if not isinstance(sweep_results, list):
        raise TypeError("sweep_results must be a list.")

    candidates: list[dict[str, float | int]] = []
    for row in sweep_results:
        if not isinstance(row, dict):
            raise TypeError("each sweep row must be a dictionary.")

        selective_accuracy = float(row["selective_accuracy"])
        if np.isnan(selective_accuracy):
            continue

        coverage = float(row["coverage"])
        threshold = float(row["threshold"])
        if not np.isfinite(selective_accuracy):
            continue
        if (
            coverage >= min_coverage_float
            and selective_accuracy >= target_accuracy_float
        ):
            candidates.append(row)

    if len(candidates) == 0:
        return None

    return min(candidates, key=lambda row: float(row["threshold"]))


def top_k_fallback_metrics(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    k: int = 3,
) -> dict[str, float | int]:
    """
    Evaluate whether abstained examples still contain the true label in Top-k.
    """
    validated_probabilities, validated_labels = validate_probabilities_and_labels(
        probabilities,
        y_true,
    )
    threshold_float = validate_confidence_threshold(threshold)
    n_classes = validated_probabilities.shape[1]

    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        raise TypeError("k must be an integer.")
    if k < 1 or k > n_classes:
        raise ValueError("k must satisfy 1 <= k <= n_classes.")
    validated_k = int(k)

    predictions = np.argmax(validated_probabilities, axis=1)
    confidences = np.max(validated_probabilities, axis=1)
    correctness = predictions == validated_labels
    answered_mask = confidences >= threshold_float
    abstained_mask = ~answered_mask

    top_k_indices = np.argsort(-validated_probabilities, axis=1)[:, :validated_k]
    abstained_top_k_hits = np.any(
        top_k_indices[abstained_mask] == validated_labels[abstained_mask, None],
        axis=1,
    )

    n_samples = int(validated_labels.shape[0])
    abstained_count = int(np.sum(abstained_mask))
    answered_count = int(np.sum(answered_mask))
    hit_count = int(np.sum(abstained_top_k_hits))

    return {
        "n_samples": n_samples,
        "threshold": threshold_float,
        "k": validated_k,
        "abstained_count": abstained_count,
        "abstained_rate": float(abstained_count / n_samples),
        "abstained_top_k_hits": hit_count,
        "abstained_top_k_hit_rate": (
            float(hit_count / abstained_count)
            if abstained_count > 0
            else float(np.nan)
        ),
        "answered_count": answered_count,
        "answered_top1_accuracy": (
            float(np.mean(correctness[answered_mask]))
            if answered_count > 0
            else float(np.nan)
        ),
    }
