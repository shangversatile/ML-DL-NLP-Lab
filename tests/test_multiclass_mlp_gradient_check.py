"""Tests for generic gradient checking with the multiclass MLP."""

import numpy as np
import pytest

from src.models.multiclass_mlp import MulticlassMLPScratch
from src.utils.model_gradient_check import (
    compare_gradients,
    compute_numerical_gradients,
    relative_l2_error,
)


def create_multiclass_gradient_check_case():
    model = MulticlassMLPScratch(
        n_features=2,
        hidden_dim=2,
        num_classes=3,
        seed=42,
    )

    model.W1 = np.array(
        [
            [1.0, -1.0],
            [0.5, 2.0],
        ]
    )
    model.b1 = np.array([0.1, -0.2])
    model.W2 = np.array(
        [
            [0.7, -0.3, 0.2],
            [-1.2, 0.4, 0.5],
        ]
    )
    model.b2 = np.array([0.05, -0.10, 0.20])

    X = np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [2.0, -0.5],
        ]
    )

    y = np.array([0, 1, 2])

    return model, X, y


def test_numerical_gradients_match_analytical_gradients() -> None:
    model, X, y = create_multiclass_gradient_check_case()

    analytical_gradients = model.compute_gradients(X, y)
    numerical_gradients = compute_numerical_gradients(model, X, y, epsilon=1e-5)
    errors = compare_gradients(analytical_gradients, numerical_gradients)

    assert all(error < 1e-6 for error in errors.values())


def test_model_parameters_restored_after_gradient_check() -> None:
    model, X, y = create_multiclass_gradient_check_case()
    original_parameters = model.get_parameters()

    compute_numerical_gradients(model, X, y)

    for name, original_parameter in original_parameters.items():
        np.testing.assert_array_equal(getattr(model, name), original_parameter)


def test_numerical_gradient_keys_and_shapes() -> None:
    model, X, y = create_multiclass_gradient_check_case()

    analytical_gradients = model.compute_gradients(X, y)
    numerical_gradients = compute_numerical_gradients(model, X, y)

    assert set(numerical_gradients) == {"dW1", "db1", "dW2", "db2"}
    for gradient_key, analytical_gradient in analytical_gradients.items():
        assert numerical_gradients[gradient_key].shape == analytical_gradient.shape


def test_relative_error_identical_arrays_is_zero() -> None:
    gradient = np.array([[1.0, -2.0], [3.0, 4.0]])

    error = relative_l2_error(gradient, gradient.copy())

    assert error == pytest.approx(0.0)


def test_relative_error_shape_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError):
        relative_l2_error(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))


def test_relative_error_non_array_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        relative_l2_error([1.0, 2.0], np.array([1.0, 2.0]))


def test_relative_error_non_finite_input_raises_value_error() -> None:
    with pytest.raises(ValueError):
        relative_l2_error(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_compare_gradients_mismatched_keys_raise_value_error() -> None:
    analytical_gradients = {"dW1": np.array([1.0])}
    numerical_gradients = {"db1": np.array([1.0])}

    with pytest.raises(ValueError):
        compare_gradients(analytical_gradients, numerical_gradients)


def test_compare_gradients_mismatched_shapes_raise_value_error() -> None:
    analytical_gradients = {"dW1": np.array([1.0, 2.0])}
    numerical_gradients = {"dW1": np.array([[1.0, 2.0]])}

    with pytest.raises(ValueError):
        compare_gradients(analytical_gradients, numerical_gradients)


@pytest.mark.parametrize("epsilon", [0.0, -1e-5])
def test_invalid_epsilon_value_raises_value_error(epsilon: float) -> None:
    model, X, y = create_multiclass_gradient_check_case()

    with pytest.raises(ValueError):
        compute_numerical_gradients(model, X, y, epsilon=epsilon)


@pytest.mark.parametrize("epsilon", [True, "small"])
def test_invalid_epsilon_type_raises_type_error(epsilon: object) -> None:
    model, X, y = create_multiclass_gradient_check_case()

    with pytest.raises(TypeError):
        compute_numerical_gradients(model, X, y, epsilon=epsilon)
