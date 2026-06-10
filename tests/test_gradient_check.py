import numpy as np
import pytest

from src.models.mlp import BinaryMLPScratch
from src.utils.gradient_check import (
    compare_gradients,
    compute_numerical_gradients,
    relative_l2_error,
)


def create_gradient_check_case():
    model = BinaryMLPScratch(n_features=2, hidden_dim=2, seed=42)

    model.W1 = np.array(
        [
            [1.0, -1.0],
            [0.5, 2.0],
        ]
    )
    model.b1 = np.array([0.1, -0.2])

    model.W2 = np.array(
        [
            [0.7],
            [-1.2],
        ]
    )
    model.b2 = np.array([0.05])

    X = np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [2.0, -0.5],
        ]
    )
    y = np.array([1, 0, 1])

    return model, X, y


def test_numerical_gradient_shapes() -> None:
    model, X, y = create_gradient_check_case()

    numerical_gradients = compute_numerical_gradients(model, X, y)

    assert set(numerical_gradients) == {"W1", "b1", "W2", "b2"}
    assert numerical_gradients["W1"].shape == model.W1.shape
    assert numerical_gradients["b1"].shape == model.b1.shape
    assert numerical_gradients["W2"].shape == model.W2.shape
    assert numerical_gradients["b2"].shape == model.b2.shape


def test_numerical_gradient_checking_agreement() -> None:
    model, X, y = create_gradient_check_case()

    analytical_gradients = model.compute_gradients(X, y)
    numerical_gradients = compute_numerical_gradients(model, X, y)
    errors = compare_gradients(analytical_gradients, numerical_gradients)

    assert all(error < 1e-6 for error in errors.values())


def test_model_parameters_restored() -> None:
    model, X, y = create_gradient_check_case()
    original_parameters = {
        "W1": model.W1.copy(),
        "b1": model.b1.copy(),
        "W2": model.W2.copy(),
        "b2": model.b2.copy(),
    }

    compute_numerical_gradients(model, X, y)

    assert np.array_equal(model.W1, original_parameters["W1"])
    assert np.array_equal(model.b1, original_parameters["b1"])
    assert np.array_equal(model.W2, original_parameters["W2"])
    assert np.array_equal(model.b2, original_parameters["b2"])


def test_relative_error_identical_arrays() -> None:
    gradient = np.array([[1.0, -2.0], [3.0, 4.0]])

    error = relative_l2_error(gradient, gradient.copy())

    assert error == pytest.approx(0.0)


def test_relative_error_shape_mismatch() -> None:
    analytical_gradient = np.array([1.0, 2.0])
    numerical_gradient = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError):
        relative_l2_error(analytical_gradient, numerical_gradient)


@pytest.mark.parametrize("epsilon", [0, -1e-5, True])
def test_invalid_epsilon(epsilon) -> None:
    model, X, y = create_gradient_check_case()

    with pytest.raises((TypeError, ValueError)):
        compute_numerical_gradients(model, X, y, epsilon=epsilon)


def test_mismatched_dictionary_keys() -> None:
    analytical_gradients = {"W1": np.array([1.0])}
    numerical_gradients = {"b1": np.array([1.0])}

    with pytest.raises(ValueError):
        compare_gradients(analytical_gradients, numerical_gradients)
