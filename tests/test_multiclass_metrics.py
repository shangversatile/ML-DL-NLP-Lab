import numpy as np
import pytest

from src.evaluation.multiclass_metrics import (
    accuracy_from_confusion_matrix,
    confusion_matrix,
    macro_average,
    per_class_precision,
    per_class_recall,
    per_sample_negative_log_likelihood,
    prediction_confidence,
    top_k_accuracy,
)


def test_confusion_matrix_exact_values() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])

    matrix = confusion_matrix(y_true, y_pred, num_classes=3)

    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
    )
    np.testing.assert_array_equal(matrix, expected)


def test_accuracy_from_confusion_matrix() -> None:
    matrix = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
    )

    assert accuracy_from_confusion_matrix(matrix) == pytest.approx(3 / 6)


def test_per_class_recall() -> None:
    matrix = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
    )

    np.testing.assert_allclose(per_class_recall(matrix), np.array([0.5, 0.5, 0.5]))


def test_per_class_precision() -> None:
    matrix = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
    )

    np.testing.assert_allclose(per_class_precision(matrix), np.array([0.5, 0.5, 0.5]))


def test_per_class_metrics_return_nan_for_zero_support() -> None:
    matrix = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ]
    )

    recall = per_class_recall(matrix)
    precision = per_class_precision(matrix)

    assert np.isnan(recall[1])
    assert np.isnan(precision[2])


def test_macro_average_ignores_nan() -> None:
    values = np.array([1.0, np.nan, 0.5])

    assert macro_average(values) == pytest.approx(0.75)


def test_top_k_accuracy() -> None:
    y_true = np.array([0, 2, 1])
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.35, 0.25],
            [0.2, 0.6, 0.2],
        ]
    )

    assert top_k_accuracy(y_true, probabilities, k=1) == pytest.approx(2 / 3)
    assert top_k_accuracy(y_true, probabilities, k=2) == pytest.approx(2 / 3)


def test_prediction_confidence() -> None:
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.35, 0.25],
            [0.2, 0.6, 0.2],
        ]
    )

    np.testing.assert_allclose(
        prediction_confidence(probabilities),
        np.array([0.7, 0.4, 0.6]),
    )


def test_per_sample_negative_log_likelihood() -> None:
    y_true = np.array([0, 2, 1])
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.35, 0.25],
            [0.2, 0.6, 0.2],
        ]
    )

    expected = -np.log(np.array([0.7, 0.25, 0.6]))

    np.testing.assert_allclose(
        per_sample_negative_log_likelihood(y_true, probabilities),
        expected,
    )


@pytest.mark.parametrize("num_classes", [1, 0, -1])
def test_confusion_matrix_invalid_num_classes_value(num_classes: int) -> None:
    with pytest.raises(ValueError):
        confusion_matrix(
            np.array([0, 1]),
            np.array([0, 1]),
            num_classes=num_classes,
        )


@pytest.mark.parametrize("num_classes", [True, 2.0, "2"])
def test_confusion_matrix_invalid_num_classes_type(num_classes: object) -> None:
    with pytest.raises(TypeError):
        confusion_matrix(
            np.array([0, 1]),
            np.array([0, 1]),
            num_classes=num_classes,
        )


def test_confusion_matrix_rejects_mismatched_sample_counts() -> None:
    with pytest.raises(ValueError):
        confusion_matrix(
            np.array([0, 1]),
            np.array([0]),
            num_classes=2,
        )


def test_confusion_matrix_rejects_invalid_labels() -> None:
    with pytest.raises(TypeError):
        confusion_matrix([0, 1], np.array([0, 1]), num_classes=2)

    with pytest.raises(ValueError):
        confusion_matrix(
            np.array([[0, 1]]),
            np.array([0, 1]),
            num_classes=2,
        )

    with pytest.raises(ValueError):
        confusion_matrix(
            np.array([0.0, 1.0]),
            np.array([0, 1]),
            num_classes=2,
        )

    with pytest.raises(ValueError):
        confusion_matrix(
            np.array([0, 2]),
            np.array([0, 1]),
            num_classes=2,
        )


def test_probability_metrics_reject_invalid_probability_matrix() -> None:
    y_true = np.array([0, 1])

    with pytest.raises(TypeError):
        top_k_accuracy(y_true, [[0.5, 0.5], [0.4, 0.6]], k=1)

    with pytest.raises(ValueError):
        top_k_accuracy(y_true, np.array([0.5, 0.5]), k=1)

    with pytest.raises(ValueError):
        top_k_accuracy(y_true, np.array([[0.5, 0.5], [0.4, np.nan]]), k=1)

    with pytest.raises(ValueError):
        top_k_accuracy(y_true, np.array([[0.5, 0.5], [-0.1, 1.1]]), k=1)

    with pytest.raises(ValueError):
        top_k_accuracy(y_true, np.array([[0.5, 0.4], [0.4, 0.6]]), k=1)


@pytest.mark.parametrize("k,expected_error", [(0, ValueError), (3, ValueError), (True, TypeError), (1.5, TypeError)])
def test_top_k_accuracy_rejects_invalid_k(
    k: object,
    expected_error: type[Exception],
) -> None:
    y_true = np.array([0, 1])
    probabilities = np.array([[0.6, 0.4], [0.4, 0.6]])

    with pytest.raises(expected_error):
        top_k_accuracy(y_true, probabilities, k=k)


def test_top_k_accuracy_rejects_mismatched_probability_rows() -> None:
    with pytest.raises(ValueError):
        top_k_accuracy(
            np.array([0]),
            np.array([[0.6, 0.4], [0.4, 0.6]]),
            k=1,
        )


def test_confusion_matrix_metrics_reject_invalid_matrix() -> None:
    with pytest.raises(TypeError):
        accuracy_from_confusion_matrix([[1, 0], [0, 1]])

    with pytest.raises(ValueError):
        accuracy_from_confusion_matrix(np.array([[1, 0, 0], [0, 1, 0]]))

    with pytest.raises(ValueError):
        accuracy_from_confusion_matrix(np.array([[1, -1], [0, 1]]))

    with pytest.raises(ValueError):
        accuracy_from_confusion_matrix(np.array([[1, np.inf], [0, 1]]))

    with pytest.raises(ValueError):
        accuracy_from_confusion_matrix(np.zeros((2, 2)))


def test_macro_average_rejects_invalid_values() -> None:
    with pytest.raises(TypeError):
        macro_average([1.0, 0.5])

    with pytest.raises(ValueError):
        macro_average(np.array([[1.0, 0.5]]))

    with pytest.raises(ValueError):
        macro_average(np.array([np.nan, np.nan]))

    with pytest.raises(ValueError):
        macro_average(np.array([1.0, np.inf]))


@pytest.mark.parametrize("epsilon,expected_error", [(0.0, ValueError), (-1.0, ValueError), (True, TypeError), ("1e-15", TypeError)])
def test_per_sample_negative_log_likelihood_rejects_invalid_epsilon(
    epsilon: object,
    expected_error: type[Exception],
) -> None:
    y_true = np.array([0, 1])
    probabilities = np.array([[0.6, 0.4], [0.4, 0.6]])

    with pytest.raises(expected_error):
        per_sample_negative_log_likelihood(
            y_true,
            probabilities,
            epsilon=epsilon,
        )
