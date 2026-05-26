"""Tests for batch gradient descent."""

import numpy as np
import pytest

from src.optimization.gradient_descent import BatchGradientDescent


def test_initialization() -> None:
    optimizer = BatchGradientDescent(learning_rate=0.1)

    assert optimizer.learning_rate == 0.1


def test_invalid_learning_rate() -> None:
    with pytest.raises(ValueError):
        BatchGradientDescent(learning_rate=0)

    with pytest.raises(ValueError):
        BatchGradientDescent(learning_rate=-0.1)

    with pytest.raises(TypeError):
        BatchGradientDescent(learning_rate="0.1")


def test_step_updates_values() -> None:
    optimizer = BatchGradientDescent(learning_rate=0.1)
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([0.1, -0.2])
    db = 0.3

    new_weights, new_bias = optimizer.step(weights, bias, dw, db)

    assert np.allclose(new_weights, np.array([0.99, 2.02]))
    assert new_bias == pytest.approx(0.47)


def test_step_does_not_modify_original_weights() -> None:
    optimizer = BatchGradientDescent(learning_rate=0.1)
    weights = np.array([1.0, 2.0])
    original_weights = weights.copy()
    dw = np.array([0.1, -0.2])

    optimizer.step(weights, bias=0.5, weight_gradients=dw, bias_gradient=0.3)

    assert np.allclose(weights, original_weights)


def test_shape_mismatch() -> None:
    optimizer = BatchGradientDescent(learning_rate=0.1)
    weights = np.array([1.0, 2.0])
    dw = np.array([0.1, -0.2, 0.3])

    with pytest.raises(ValueError):
        optimizer.step(weights, bias=0.5, weight_gradients=dw, bias_gradient=0.3)
