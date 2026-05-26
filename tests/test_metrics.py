"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import mean_squared_error


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
