"""Tests for multiclass probability and loss utilities."""

import numpy as np
import pytest

from src.utils.multiclass import (
    multiclass_cross_entropy,
    multiclass_cross_entropy_from_logits,
    one_hot_encode,
    softmax_cross_entropy_gradient,
    stable_softmax,
)


def test_one_hot_encode_shape_values_dtype_and_no_mutation() -> None:
    y = np.array([0, 2, 1])
    original = y.copy()

    encoded = one_hot_encode(y, num_classes=3)

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert encoded.shape == (3, 3)
    np.testing.assert_array_equal(encoded, expected)
    assert encoded.dtype == float
    np.testing.assert_array_equal(y, original)


def test_one_hot_encode_empty_labels_raise_value_error() -> None:
    with pytest.raises(ValueError):
        one_hot_encode(np.array([], dtype=int), num_classes=3)


def test_one_hot_encode_non_array_labels_raise_type_error() -> None:
    with pytest.raises(TypeError):
        one_hot_encode([0, 1], num_classes=2)


def test_one_hot_encode_non_1d_labels_raise_value_error() -> None:
    with pytest.raises(ValueError):
        one_hot_encode(np.array([[0, 1]]), num_classes=2)


def test_one_hot_encode_non_integer_labels_raise_value_error() -> None:
    with pytest.raises(ValueError):
        one_hot_encode(np.array([0.0, 1.0]), num_classes=2)


def test_one_hot_encode_negative_labels_raise_value_error() -> None:
    with pytest.raises(ValueError):
        one_hot_encode(np.array([0, -1]), num_classes=2)


def test_one_hot_encode_too_large_labels_raise_value_error() -> None:
    with pytest.raises(ValueError):
        one_hot_encode(np.array([0, 2]), num_classes=2)


@pytest.mark.parametrize("num_classes", [0, -1])
def test_one_hot_encode_non_positive_num_classes_raise_value_error(
    num_classes: int,
) -> None:
    with pytest.raises(ValueError):
        one_hot_encode(np.array([0]), num_classes=num_classes)


@pytest.mark.parametrize("num_classes", [True, 2.0])
def test_one_hot_encode_invalid_num_classes_type_raises_type_error(
    num_classes: object,
) -> None:
    with pytest.raises(TypeError):
        one_hot_encode(np.array([0]), num_classes=num_classes)


def test_stable_softmax_shape_row_sums_and_non_negative_values() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 1.0]])

    probabilities = stable_softmax(logits)

    assert probabilities.shape == logits.shape
    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))
    assert np.all(probabilities >= 0.0)


def test_stable_softmax_uniform_logits_give_uniform_probabilities() -> None:
    logits = np.array([[5.0, 5.0, 5.0]])

    probabilities = stable_softmax(logits)

    np.testing.assert_allclose(probabilities, np.array([[1 / 3, 1 / 3, 1 / 3]]))


def test_stable_softmax_is_shift_invariant() -> None:
    logits = np.array([[1.0, 2.0, 3.0]])
    shifted = logits + 100.0

    probabilities = stable_softmax(logits)
    shifted_probabilities = stable_softmax(shifted)

    np.testing.assert_allclose(probabilities, shifted_probabilities)


def test_stable_softmax_handles_very_large_logits() -> None:
    logits = np.array([[1000.0, 1001.0, 1002.0]])

    probabilities = stable_softmax(logits)

    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.array([1.0]))


def test_stable_softmax_non_array_logits_raise_type_error() -> None:
    with pytest.raises(TypeError):
        stable_softmax([[1.0, 2.0]])


def test_stable_softmax_non_2d_logits_raise_value_error() -> None:
    with pytest.raises(ValueError):
        stable_softmax(np.array([1.0, 2.0]))


def test_stable_softmax_non_finite_logits_raise_value_error() -> None:
    with pytest.raises(ValueError):
        stable_softmax(np.array([[1.0, np.inf]]))


def test_multiclass_cross_entropy_exact_value() -> None:
    y = np.array([0, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )
    expected = -np.mean(
        [
            np.log(0.8),
            np.log(0.5),
        ]
    )

    loss = multiclass_cross_entropy(y, probabilities)

    assert loss == pytest.approx(expected)


def test_multiclass_cross_entropy_perfect_confident_predictions_have_low_loss() -> None:
    y = np.array([0, 1, 2])
    probabilities = np.eye(3)

    loss = multiclass_cross_entropy(y, probabilities)

    assert loss < 1e-12


def test_multiclass_cross_entropy_wrong_confident_predictions_have_high_loss() -> None:
    y = np.array([0])
    probabilities = np.array([[0.0, 1.0, 0.0]])

    loss = multiclass_cross_entropy(y, probabilities)

    assert loss > 30.0


def test_multiclass_cross_entropy_invalid_shapes_raise_errors() -> None:
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])

    with pytest.raises(ValueError):
        multiclass_cross_entropy(np.array([[0, 1]]), probabilities)

    with pytest.raises(ValueError):
        multiclass_cross_entropy(np.array([0, 1]), np.array([0.8, 0.2]))

    with pytest.raises(ValueError):
        multiclass_cross_entropy(np.array([0]), probabilities)


def test_multiclass_cross_entropy_row_sums_not_one_raise_value_error() -> None:
    with pytest.raises(ValueError):
        multiclass_cross_entropy(np.array([0]), np.array([[0.8, 0.1]]))


def test_multiclass_cross_entropy_negative_probability_raises_value_error() -> None:
    with pytest.raises(ValueError):
        multiclass_cross_entropy(np.array([0]), np.array([[1.1, -0.1]]))


def test_multiclass_cross_entropy_non_finite_probability_raises_value_error() -> None:
    with pytest.raises(ValueError):
        multiclass_cross_entropy(np.array([0]), np.array([[np.nan, 1.0]]))


@pytest.mark.parametrize("epsilon", [0.0, -1e-15])
def test_multiclass_cross_entropy_non_positive_epsilon_raises_value_error(
    epsilon: float,
) -> None:
    with pytest.raises(ValueError):
        multiclass_cross_entropy(np.array([0]), np.array([[1.0, 0.0]]), epsilon)


@pytest.mark.parametrize("epsilon", [True, "small"])
def test_multiclass_cross_entropy_invalid_epsilon_type_raises_type_error(
    epsilon: object,
) -> None:
    with pytest.raises(TypeError):
        multiclass_cross_entropy(np.array([0]), np.array([[1.0, 0.0]]), epsilon)


def test_multiclass_cross_entropy_from_logits_matches_probability_version() -> None:
    y = np.array([0, 2])
    logits = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
        ]
    )
    probabilities = stable_softmax(logits)

    loss_a = multiclass_cross_entropy(y, probabilities)
    loss_b = multiclass_cross_entropy_from_logits(y, logits)

    assert loss_a == pytest.approx(loss_b)


def test_softmax_cross_entropy_gradient_exact_values_shape_and_row_sums() -> None:
    y = np.array([0, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )
    expected = np.array(
        [
            [-0.1, 0.05, 0.05],
            [0.1, 0.15, -0.25],
        ]
    )

    gradient = softmax_cross_entropy_gradient(y, probabilities)

    np.testing.assert_allclose(gradient, expected)
    assert gradient.shape == probabilities.shape
    np.testing.assert_allclose(np.sum(gradient, axis=1), np.zeros(2), atol=1e-15)


def test_softmax_cross_entropy_gradient_invalid_inputs_raise_errors() -> None:
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])

    with pytest.raises(TypeError):
        softmax_cross_entropy_gradient([0, 1], probabilities)

    with pytest.raises(ValueError):
        softmax_cross_entropy_gradient(np.array([0]), probabilities)

    with pytest.raises(ValueError):
        softmax_cross_entropy_gradient(np.array([0, 1]), np.array([[0.8, 0.1]]))
