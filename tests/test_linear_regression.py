"""Tests for linear regression."""

import numpy as np
import pytest

from src.models.linear_regression import LinearRegressionScratch


def test_initialization() -> None:
    model = LinearRegressionScratch(n_features=3)

    assert model.weights.shape == (3,)
    assert np.allclose(model.weights, np.zeros(3))
    assert model.bias == 0.0


def test_predict_shape() -> None:
    model = LinearRegressionScratch(n_features=3)
    X = np.ones((5, 3))

    predictions = model.predict(X)

    assert predictions.shape == (5,)


def test_predict_values_with_manual_parameters() -> None:
    model = LinearRegressionScratch(n_features=2)
    model.weights = np.array([1.0, 2.0])
    model.bias = 0.5
    X = np.array([[1.0, 1.0], [2.0, 0.0]])

    predictions = model.predict(X)

    assert np.allclose(predictions, np.array([3.5, 2.5]))


def test_compute_loss() -> None:
    model = LinearRegressionScratch(n_features=1)
    model.weights = np.array([2.0])
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 5.0, 7.0])

    loss = model.compute_loss(X, y)

    assert loss == pytest.approx(2 / 3)


def test_compute_gradients() -> None:
    model = LinearRegressionScratch(n_features=1)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    dw, db = model.compute_gradients(X, y)

    assert np.allclose(dw, np.array([-56 / 3]))
    assert db == pytest.approx(-8.0)


def test_invalid_n_features() -> None:
    with pytest.raises(ValueError):
        LinearRegressionScratch(n_features=0)
