"""Tests for Adam optimizer."""

import numpy as np
import pytest

from src.optimization.adam import Adam


def test_initialization() -> None:
    optimizer = Adam()

    assert optimizer.learning_rate == 0.001
    assert optimizer.beta1 == 0.9
    assert optimizer.beta2 == 0.999
    assert optimizer.epsilon == 1e-8
    assert optimizer.first_moment_weights is None
    assert optimizer.second_moment_weights is None
    assert optimizer.first_moment_bias == 0.0
    assert optimizer.second_moment_bias == 0.0
    assert optimizer.time_step == 0


def test_invalid_learning_rate() -> None:
    with pytest.raises(ValueError):
        Adam(learning_rate=0)

    with pytest.raises(ValueError):
        Adam(learning_rate=-0.1)

    with pytest.raises(TypeError):
        Adam(learning_rate="0.001")


def test_invalid_beta1() -> None:
    with pytest.raises(ValueError):
        Adam(beta1=-0.1)

    with pytest.raises(ValueError):
        Adam(beta1=1.0)

    with pytest.raises(TypeError):
        Adam(beta1="0.9")


def test_invalid_beta2() -> None:
    with pytest.raises(ValueError):
        Adam(beta2=-0.1)

    with pytest.raises(ValueError):
        Adam(beta2=1.0)

    with pytest.raises(TypeError):
        Adam(beta2="0.999")


def test_invalid_epsilon() -> None:
    with pytest.raises(ValueError):
        Adam(epsilon=0)

    with pytest.raises(ValueError):
        Adam(epsilon=-1e-8)

    with pytest.raises(TypeError):
        Adam(epsilon="1e-8")


def test_first_step_updates_values_and_state() -> None:
    optimizer = Adam(
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([0.1, -0.2])
    db = 0.3

    new_weights, new_bias = optimizer.step(weights, bias, dw, db)

    assert np.allclose(optimizer.first_moment_weights, np.array([0.01, -0.02]))
    assert optimizer.first_moment_bias == pytest.approx(0.03)
    assert np.allclose(
        optimizer.second_moment_weights,
        np.array([0.00001, 0.00004]),
    )
    assert optimizer.second_moment_bias == pytest.approx(0.00009)
    assert np.allclose(new_weights, np.array([0.9, 2.1]))
    assert new_bias == pytest.approx(0.4)
    assert isinstance(new_bias, float)
    assert optimizer.time_step == 1


def test_second_step_continues_in_same_direction() -> None:
    optimizer = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([0.1, -0.2])
    db = 0.3

    first_weights, first_bias = optimizer.step(weights, bias, dw, db)
    second_weights, second_bias = optimizer.step(first_weights, first_bias, dw, db)

    assert optimizer.time_step == 2
    assert second_weights[0] < first_weights[0]
    assert second_weights[1] > first_weights[1]
    assert second_bias < first_bias


def test_step_does_not_modify_original_weights() -> None:
    optimizer = Adam(learning_rate=0.1)
    weights = np.array([1.0, 2.0])
    original_weights = weights.copy()
    dw = np.array([0.1, -0.2])

    optimizer.step(weights, bias=0.5, weight_gradients=dw, bias_gradient=0.3)

    assert np.allclose(weights, original_weights)


def test_shape_mismatch() -> None:
    optimizer = Adam(learning_rate=0.1)
    weights = np.array([1.0, 2.0])
    dw = np.array([0.1, -0.2, 0.3])

    with pytest.raises(ValueError):
        optimizer.step(weights, bias=0.5, weight_gradients=dw, bias_gradient=0.3)
