"""Tests for selective prediction and abstention diagnostics."""

import numpy as np
import pytest

from src.evaluation.selective_prediction import (
    abstention_mask_from_confidence,
    choose_threshold_for_target_selective_accuracy,
    selective_prediction_metrics,
    threshold_sweep,
    top_k_fallback_metrics,
    top_label_predictions,
    validate_confidence_threshold,
)


def test_validate_confidence_threshold_accepts_valid_values() -> None:
    assert validate_confidence_threshold(0.0) == 0.0
    assert validate_confidence_threshold(0.5) == 0.5
    assert validate_confidence_threshold(1.0) == 1.0
    assert validate_confidence_threshold(np.float64(0.25)) == 0.25


def test_validate_confidence_threshold_rejects_invalid_values() -> None:
    with pytest.raises(TypeError):
        validate_confidence_threshold(True)

    with pytest.raises(ValueError):
        validate_confidence_threshold(-0.1)

    with pytest.raises(ValueError):
        validate_confidence_threshold(1.1)

    with pytest.raises(ValueError):
        validate_confidence_threshold(float("nan"))

    with pytest.raises(ValueError):
        validate_confidence_threshold(float("inf"))


def test_top_label_predictions_returns_predictions_confidences_and_correctness() -> None:
    probabilities = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.6, 0.3, 0.1],
            [0.2, 0.3, 0.5],
        ],
    )
    y_true = np.array([1, 2, 0])

    predictions, confidences, correctness = top_label_predictions(
        probabilities,
        y_true,
    )

    np.testing.assert_array_equal(predictions, np.array([1, 0, 2]))
    np.testing.assert_allclose(confidences, np.array([0.7, 0.6, 0.5]))
    np.testing.assert_array_equal(correctness, np.array([True, False, False]))


def test_abstention_mask_from_confidence_uses_threshold_inclusively() -> None:
    probabilities = np.array(
        [
            [0.6, 0.4],
            [0.5, 0.5],
            [0.2, 0.8],
            [1.0, 0.0],
        ],
    )

    np.testing.assert_array_equal(
        abstention_mask_from_confidence(probabilities, 0.6),
        np.array([True, False, True, True]),
    )
    np.testing.assert_array_equal(
        abstention_mask_from_confidence(probabilities, 0.0),
        np.array([True, True, True, True]),
    )
    np.testing.assert_array_equal(
        abstention_mask_from_confidence(probabilities, 1.0),
        np.array([False, False, False, True]),
    )


def test_selective_prediction_metrics_for_deterministic_example() -> None:
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.55, 0.45],
            [0.51, 0.49],
        ],
    )
    y_true = np.array([0, 1, 0, 1])

    metrics = selective_prediction_metrics(probabilities, y_true, threshold=0.7)

    assert metrics["n_samples"] == 4
    assert metrics["answered_count"] == 2
    assert metrics["abstained_count"] == 2
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["abstention_rate"] == pytest.approx(0.5)
    assert metrics["original_accuracy"] == pytest.approx(0.5)
    assert metrics["selective_accuracy"] == pytest.approx(0.5)
    assert metrics["selective_error_rate"] == pytest.approx(0.5)
    assert metrics["errors_total"] == 2
    assert metrics["errors_answered"] == 1
    assert metrics["errors_abstained"] == 1
    assert metrics["error_abstention_rate"] == pytest.approx(0.5)
    assert metrics["mean_answered_confidence"] == pytest.approx(0.85)
    assert metrics["mean_abstained_confidence"] == pytest.approx(0.53)


def test_selective_prediction_metrics_preserves_nan_for_empty_answered_group() -> None:
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2]])
    y_true = np.array([0, 1])

    metrics = selective_prediction_metrics(probabilities, y_true, threshold=1.0)

    assert metrics["answered_count"] == 0
    assert np.isnan(metrics["selective_accuracy"])
    assert np.isnan(metrics["selective_error_rate"])
    assert np.isnan(metrics["mean_answered_confidence"])


def test_selective_prediction_metrics_preserves_nan_when_no_errors_exist() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    y_true = np.array([0, 1])

    metrics = selective_prediction_metrics(probabilities, y_true, threshold=0.5)

    assert metrics["errors_total"] == 0
    assert np.isnan(metrics["error_abstention_rate"])


def test_threshold_sweep_default_length_is_twenty_one() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    y_true = np.array([0, 1])

    results = threshold_sweep(probabilities, y_true)

    assert len(results) == 21
    assert results[0]["threshold"] == pytest.approx(0.0)
    assert results[-1]["threshold"] == pytest.approx(1.0)


def test_threshold_sweep_preserves_custom_threshold_order() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    y_true = np.array([0, 1])

    results = threshold_sweep(probabilities, y_true, thresholds=[0.9, 0.0, 0.5])

    assert [row["threshold"] for row in results] == [0.9, 0.0, 0.5]


def test_threshold_sweep_rejects_invalid_threshold() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    y_true = np.array([0, 1])

    with pytest.raises(ValueError):
        threshold_sweep(probabilities, y_true, thresholds=[0.5, -0.1])


def test_choose_threshold_for_target_selective_accuracy_returns_lowest_threshold() -> None:
    sweep_results = [
        {"threshold": 0.7, "selective_accuracy": 0.96, "coverage": 0.4},
        {"threshold": 0.5, "selective_accuracy": 0.95, "coverage": 0.8},
        {"threshold": 0.9, "selective_accuracy": np.nan, "coverage": 0.0},
        {"threshold": 0.0, "selective_accuracy": 0.8, "coverage": 1.0},
    ]

    selected = choose_threshold_for_target_selective_accuracy(
        sweep_results,
        target_accuracy=0.95,
        min_coverage=0.5,
    )

    assert selected is not None
    assert selected["threshold"] == 0.5


def test_choose_threshold_for_target_selective_accuracy_returns_none_when_impossible() -> None:
    sweep_results = [
        {"threshold": 0.0, "selective_accuracy": 0.8, "coverage": 1.0},
        {"threshold": 0.5, "selective_accuracy": 0.9, "coverage": 0.8},
    ]

    assert (
        choose_threshold_for_target_selective_accuracy(
            sweep_results,
            target_accuracy=0.99,
            min_coverage=0.5,
        )
        is None
    )


def test_choose_threshold_for_target_selective_accuracy_rejects_invalid_constraints() -> None:
    sweep_results = [
        {"threshold": 0.0, "selective_accuracy": 0.8, "coverage": 1.0},
    ]

    with pytest.raises(ValueError):
        choose_threshold_for_target_selective_accuracy(
            sweep_results,
            target_accuracy=-0.1,
        )

    with pytest.raises(ValueError):
        choose_threshold_for_target_selective_accuracy(
            sweep_results,
            target_accuracy=0.9,
            min_coverage=1.1,
        )


def test_top_k_fallback_metrics_validates_k() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    y_true = np.array([0, 1])

    with pytest.raises(TypeError):
        top_k_fallback_metrics(probabilities, y_true, threshold=0.5, k=True)

    with pytest.raises(ValueError):
        top_k_fallback_metrics(probabilities, y_true, threshold=0.5, k=0)

    with pytest.raises(ValueError):
        top_k_fallback_metrics(probabilities, y_true, threshold=0.5, k=3)


def test_top_k_fallback_metrics_reports_abstained_top_k_hit_rate() -> None:
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1, 0.0],
            [0.45, 0.4, 0.1, 0.05],
            [0.4, 0.35, 0.2, 0.05],
            [0.9, 0.05, 0.03, 0.02],
        ],
    )
    y_true = np.array([0, 1, 2, 1])

    metrics = top_k_fallback_metrics(probabilities, y_true, threshold=0.5, k=2)

    assert metrics["n_samples"] == 4
    assert metrics["k"] == 2
    assert metrics["abstained_count"] == 2
    assert metrics["abstained_rate"] == pytest.approx(0.5)
    assert metrics["abstained_top_k_hits"] == 1
    assert metrics["abstained_top_k_hit_rate"] == pytest.approx(0.5)
    assert metrics["answered_count"] == 2
    assert metrics["answered_top1_accuracy"] == pytest.approx(0.5)


def test_top_k_fallback_metrics_preserves_nan_when_no_samples_abstain() -> None:
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    y_true = np.array([0, 1])

    metrics = top_k_fallback_metrics(probabilities, y_true, threshold=0.0, k=2)

    assert metrics["abstained_count"] == 0
    assert np.isnan(metrics["abstained_top_k_hit_rate"])
