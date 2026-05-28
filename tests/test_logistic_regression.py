"""Tests for logistic regression."""

import numpy as np
import pytest

from src.models.logistic_regression import LogisticRegressionScratch


def test_initialization() -> None:
    model = LogisticRegressionScratch(n_features=3)

    assert model.weights.shape == (3,)
    assert np.all(model.weights == 0.0)
    assert model.bias == 0.0


def test_sigmoid() -> None:
    model = LogisticRegressionScratch(n_features=1)

    probabilities = model._sigmoid(np.array([0.0, 1000.0, -1000.0]))

    assert probabilities[0] == pytest.approx(0.5)
    assert probabilities[1] == pytest.approx(1.0)
    assert probabilities[2] == pytest.approx(0.0)


def test_predict_proba_shape_and_values() -> None:
    model = LogisticRegressionScratch(n_features=2)
    model.weights = np.array([1.0, -1.0])
    model.bias = 0.0
    X = np.array([[1.0, 1.0], [2.0, 0.0]])

    probabilities = model.predict_proba(X)

    expected = np.array([0.5, 1.0 / (1.0 + np.exp(-2.0))])
    assert probabilities.shape == (2,)
    assert np.allclose(probabilities, expected)


def test_predict_labels() -> None:
    model = LogisticRegressionScratch(n_features=2)
    model.weights = np.array([1.0, -1.0])
    model.bias = 0.0
    X = np.array([[1.0, 1.0], [2.0, 0.0]])

    labels = model.predict(X, threshold=0.5)

    assert labels.dtype == int
    assert np.array_equal(labels, np.array([1, 1]))


def test_binary_cross_entropy_loss() -> None:
    model = LogisticRegressionScratch(n_features=1)
    model.weights = np.array([0.0])
    model.bias = 0.0
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])

    loss = model.compute_loss(X, y)

    assert loss == pytest.approx(0.693147, rel=1e-6)


def test_gradients() -> None:
    model = LogisticRegressionScratch(n_features=1)
    model.weights = np.array([0.0])
    model.bias = 0.0
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])

    dw, db = model.compute_gradients(X, y)

    assert np.allclose(dw, np.array([-0.25]))
    assert db == pytest.approx(0.0)


def test_invalid_labels() -> None:
    model = LogisticRegressionScratch(n_features=1)
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 2])

    with pytest.raises(ValueError):
        model.compute_loss(X, y)


def test_invalid_n_features() -> None:
    with pytest.raises(ValueError):
        LogisticRegressionScratch(n_features=0)
