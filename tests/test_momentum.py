"""Tests for Momentum optimizer."""

import numpy as np
import pytest

from src.optimization.momentum import Momentum


def test_initialization() -> None:
    optimizer = Momentum(learning_rate=0.1, beta=0.9)

    assert optimizer.learning_rate == 0.1
    assert optimizer.beta == 0.9
    assert optimizer.velocity_weights is None
    assert optimizer.velocity_bias == 0.0


def test_invalid_learning_rate() -> None:
    with pytest.raises(ValueError):
        Momentum(learning_rate=0)

    with pytest.raises(ValueError):
        Momentum(learning_rate=-0.1)

    with pytest.raises(TypeError):
        Momentum(learning_rate="0.1")


def test_invalid_beta() -> None:
    with pytest.raises(ValueError):
        Momentum(beta=-0.1)

    with pytest.raises(ValueError):
        Momentum(beta=1.0)

    with pytest.raises(TypeError):
        Momentum(beta="0.9")


def test_first_step_updates_values_and_state() -> None:
    optimizer = Momentum(learning_rate=0.1, beta=0.9)
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([0.1, -0.2])
    db = 0.3

    new_weights, new_bias = optimizer.step(weights, bias, dw, db)

    assert np.allclose(optimizer.velocity_weights, np.array([0.01, -0.02]))
    assert optimizer.velocity_bias == pytest.approx(0.03)
    assert np.allclose(new_weights, np.array([0.999, 2.002]))
    assert new_bias == pytest.approx(0.497)


def test_second_step_accumulates_state() -> None:
    optimizer = Momentum(learning_rate=0.1, beta=0.9)
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([0.1, -0.2])
    db = 0.3

    first_weights, first_bias = optimizer.step(weights, bias, dw, db)
    second_weights, second_bias = optimizer.step(first_weights, first_bias, dw, db)

    assert np.allclose(optimizer.velocity_weights, np.array([0.019, -0.038]))
    assert optimizer.velocity_bias == pytest.approx(0.057)
    assert np.allclose(second_weights, np.array([0.9971, 2.0058]))
    assert second_bias == pytest.approx(0.4913)


def test_step_does_not_modify_original_weights() -> None:
    optimizer = Momentum(learning_rate=0.1, beta=0.9)
    weights = np.array([1.0, 2.0])
    original_weights = weights.copy()
    dw = np.array([0.1, -0.2])

    optimizer.step(weights, bias=0.5, weight_gradients=dw, bias_gradient=0.3)

    assert np.allclose(weights, original_weights)


def test_shape_mismatch() -> None:
    optimizer = Momentum(learning_rate=0.1, beta=0.9)
    weights = np.array([1.0, 2.0])
    dw = np.array([0.1, -0.2, 0.3])

    with pytest.raises(ValueError):
        optimizer.step(weights, bias=0.5, weight_gradients=dw, bias_gradient=0.3)
