"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    accuracy_score,
    binary_cross_entropy,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)


def test_mean_squared_error_exact_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 5.0])

    mse = mean_squared_error(y_true, y_pred)

    assert mse == pytest.approx(5 / 3)


def test_mean_squared_error_zero() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    mse = mean_squared_error(y_true, y_pred)

    assert mse == 0.0


def test_mean_squared_error_shape_mismatch() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        mean_squared_error(y_true, y_pred)


def test_mean_squared_error_non_1d_input() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError):
        mean_squared_error(y_true, y_pred)


def test_mean_squared_error_non_array_input() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    with pytest.raises(TypeError):
        mean_squared_error([1.0, 2.0, 3.0], y_pred)

    with pytest.raises(TypeError):
        mean_squared_error(y_true, [1.0, 2.0, 3.0])


def test_binary_cross_entropy_exact_value_with_half_probabilities() -> None:
    y_true = np.array([0, 1])
    y_prob = np.array([0.5, 0.5])

    loss = binary_cross_entropy(y_true, y_prob)

    assert loss == pytest.approx(0.693147, rel=1e-6)


def test_binary_cross_entropy_lower_for_confident_correct_predictions() -> None:
    y_true = np.array([0, 1])
    y_prob_good = np.array([0.1, 0.9])
    y_prob_bad = np.array([0.4, 0.6])

    good_loss = binary_cross_entropy(y_true, y_prob_good)
    bad_loss = binary_cross_entropy(y_true, y_prob_bad)

    assert good_loss < bad_loss


def test_binary_cross_entropy_invalid_probabilities() -> None:
    y_true = np.array([0, 1])

    with pytest.raises(ValueError):
        binary_cross_entropy(y_true, np.array([-0.1, 0.9]))

    with pytest.raises(ValueError):
        binary_cross_entropy(y_true, np.array([0.1, 1.1]))


def test_binary_cross_entropy_shape_mismatch() -> None:
    y_true = np.array([0, 1])
    y_prob = np.array([0.5])

    with pytest.raises(ValueError):
        binary_cross_entropy(y_true, y_prob)


def test_binary_cross_entropy_non_binary_y_true() -> None:
    y_true = np.array([0, 2])
    y_prob = np.array([0.5, 0.5])

    with pytest.raises(ValueError):
        binary_cross_entropy(y_true, y_prob)


def test_confusion_matrix_binary() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    matrix = confusion_matrix(y_true, y_pred)

    expected = np.array([[1, 1], [1, 2]])
    assert np.array_equal(matrix, expected)


def test_accuracy_score_binary() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    accuracy = accuracy_score(y_true, y_pred)

    assert accuracy == pytest.approx(3 / 5)


def test_precision_score_binary() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    precision = precision_score(y_true, y_pred)

    assert precision == pytest.approx(2 / 3)


def test_recall_score_binary() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    recall = recall_score(y_true, y_pred)

    assert recall == pytest.approx(2 / 3)


def test_f1_score_binary() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    score = f1_score(y_true, y_pred)

    assert score == pytest.approx(2 / 3)


def test_precision_recall_f1_zero_division() -> None:
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])

    assert precision_score(y_true, y_pred) == 0.0
    assert recall_score(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0


def test_binary_metrics_non_binary_label() -> None:
    y_true = np.array([0, 2, 1])
    y_pred = np.array([0, 1, 1])

    with pytest.raises(ValueError):
        accuracy_score(y_true, y_pred)


def test_binary_metrics_shape_mismatch() -> None:
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError):
        accuracy_score(y_true, y_pred)


def test_binary_metrics_non_1d_input() -> None:
    y_true = np.array([[0, 1], [1, 0]])
    y_pred = np.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError):
        accuracy_score(y_true, y_pred)
