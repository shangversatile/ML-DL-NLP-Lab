"""Tests for real canvas validation diagnostics."""

import numpy as np
import pytest

from src.evaluation.canvas_diagnostics import (
    canvas_confusion_matrix,
    high_confidence_canvas_errors,
    per_class_canvas_summary,
    summarize_canvas_validation,
    top_k_miss_errors,
)


def test_per_class_canvas_summary_counts_accuracy_and_empty_class() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    top_k_indices = np.array(
        [
            [0, 1],
            [2, 1],
            [1, 0],
            [2, 0],
            [1, 2],
        ]
    )
    confidences = np.array([0.90, 0.80, 0.70, 0.60, 0.95])

    summary = per_class_canvas_summary(
        y_true,
        y_pred,
        top_k_indices,
        confidences,
        num_classes=3,
    )

    assert summary[0]["count"] == 2
    assert summary[0]["correct"] == 1
    assert summary[0]["errors"] == 1
    assert summary[0]["accuracy"] == pytest.approx(0.5)
    assert summary[0]["top_k_accuracy"] == pytest.approx(0.5)
    assert summary[0]["mean_confidence"] == pytest.approx(0.85)
    assert summary[0]["mean_error_confidence"] == pytest.approx(0.80)

    assert summary[1]["count"] == 3
    assert summary[1]["correct"] == 2
    assert summary[1]["errors"] == 1
    assert summary[1]["accuracy"] == pytest.approx(2 / 3)
    assert summary[1]["top_k_accuracy"] == pytest.approx(2 / 3)
    assert summary[1]["mean_confidence"] == pytest.approx(0.75)
    assert summary[1]["mean_error_confidence"] == pytest.approx(0.60)

    assert summary[2]["count"] == 0
    assert summary[2]["correct"] == 0
    assert summary[2]["errors"] == 0
    assert np.isnan(summary[2]["accuracy"])
    assert np.isnan(summary[2]["top_k_accuracy"])
    assert np.isnan(summary[2]["mean_confidence"])
    assert np.isnan(summary[2]["mean_error_confidence"])


def test_canvas_confusion_matrix_shape_and_counts() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    matrix = canvas_confusion_matrix(y_true, y_pred, num_classes=3)

    expected = np.array(
        [
            [1, 1, 0],
            [1, 2, 0],
            [0, 0, 0],
        ]
    )
    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(matrix, expected)


def test_high_confidence_canvas_errors_filters_threshold_and_sorts() -> None:
    records = [
        {
            "path": "low_error.npz",
            "true_label": 6,
            "prediction": 5,
            "confidence": 0.89,
            "top_k_indices": np.array([5, 6, 4]),
        },
        {
            "path": "high_error.npz",
            "true_label": 8,
            "prediction": 0,
            "confidence": 0.95,
            "top_k_indices": np.array([0, 3, 9]),
        },
        {
            "path": "correct_high.npz",
            "true_label": 4,
            "prediction": 4,
            "confidence": 0.99,
            "top_k_indices": np.array([4, 9, 7]),
        },
        {
            "path": "second_error.npz",
            "true_label": 9,
            "prediction": 4,
            "confidence": 0.91,
            "top_k_indices": np.array([4, 9, 7]),
        },
    ]

    errors = high_confidence_canvas_errors(records, threshold=0.90)

    assert [record["path"] for record in errors] == [
        "high_error.npz",
        "second_error.npz",
    ]
    assert all(record["prediction"] != record["true_label"] for record in errors)
    assert all(record["confidence"] >= 0.90 for record in errors)


def test_top_k_miss_errors_filters_misses_and_sorts() -> None:
    records = [
        {
            "path": "ranked_error.npz",
            "true_label": 9,
            "prediction": 4,
            "confidence": 0.80,
            "top_k_indices": np.array([4, 9, 7]),
        },
        {
            "path": "severe_error.npz",
            "true_label": 8,
            "prediction": 0,
            "confidence": 0.95,
            "top_k_indices": np.array([0, 3, 9]),
        },
        {
            "path": "correct.npz",
            "true_label": 0,
            "prediction": 0,
            "confidence": 0.99,
            "top_k_indices": np.array([0, 8, 6]),
        },
        {
            "path": "lower_conf_miss.npz",
            "true_label": 6,
            "prediction": 5,
            "confidence": 0.90,
            "top_k_indices": np.array([5, 4, 3]),
        },
    ]

    errors = top_k_miss_errors(records)

    assert [record["path"] for record in errors] == [
        "severe_error.npz",
        "lower_conf_miss.npz",
    ]
    assert all(record["prediction"] != record["true_label"] for record in errors)
    assert all(
        record["true_label"] not in set(record["top_k_indices"].tolist())
        for record in errors
    )


def test_summarize_canvas_validation_overall_metrics() -> None:
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 0, 2, 0])
    probabilities = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.70, 0.20, 0.10],
            [0.20, 0.30, 0.50],
            [0.60, 0.30, 0.10],
        ]
    )
    top_k_indices = np.array(
        [
            [0, 1],
            [0, 1],
            [2, 1],
            [0, 1],
        ]
    )

    summary = summarize_canvas_validation(
        y_true,
        y_pred,
        probabilities,
        top_k_indices,
    )

    assert summary["n_samples"] == 4
    assert summary["n_errors"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["top_k_accuracy"] == pytest.approx(0.75)
    assert summary["mean_confidence"] == pytest.approx(0.65)
    assert summary["overconfidence_gap"] == pytest.approx(0.15)
    assert summary["top_k_miss_error_count"] == 1
    assert summary["top_k_miss_error_rate"] == pytest.approx(0.5)


def test_canvas_diagnostics_reject_invalid_inputs() -> None:
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    top_k_indices = np.array([[0, 1], [1, 0]])
    confidences = np.array([0.9, 0.8])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])

    with pytest.raises(ValueError):
        per_class_canvas_summary(
            y_true,
            np.array([0]),
            top_k_indices,
            confidences,
            num_classes=2,
        )

    with pytest.raises(ValueError):
        per_class_canvas_summary(
            y_true,
            y_pred,
            np.array([[0, 1]]),
            confidences,
            num_classes=2,
        )

    with pytest.raises(ValueError):
        canvas_confusion_matrix(y_true, y_pred, num_classes=0)

    with pytest.raises(ValueError):
        summarize_canvas_validation(
            y_true,
            y_pred,
            probabilities,
            np.array([[0, 1]]),
        )

    with pytest.raises(ValueError):
        high_confidence_canvas_errors([], threshold=-0.1)

    with pytest.raises(ValueError):
        high_confidence_canvas_errors([], threshold=1.1)

    with pytest.raises(ValueError):
        high_confidence_canvas_errors([], threshold=float("nan"))

    with pytest.raises(TypeError):
        high_confidence_canvas_errors([], threshold=True)
