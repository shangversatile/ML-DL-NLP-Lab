"""Tests for the NumPy MLP forward pass."""

import numpy as np
import pytest

from src.models.mlp import BinaryMLPScratch


def test_initialization_shapes() -> None:
    model = BinaryMLPScratch(n_features=3, hidden_dim=4)

    assert model.W1.shape == (3, 4)
    assert model.b1.shape == (4,)
    assert model.W2.shape == (4, 1)
    assert model.b2.shape == (1,)


def test_reproducible_initialization() -> None:
    model_a = BinaryMLPScratch(n_features=3, hidden_dim=4, seed=7)
    model_b = BinaryMLPScratch(n_features=3, hidden_dim=4, seed=7)

    assert np.allclose(model_a.W1, model_b.W1)
    assert np.allclose(model_a.W2, model_b.W2)


def test_different_seeds_initialize_different_weights() -> None:
    model_a = BinaryMLPScratch(n_features=3, hidden_dim=4, seed=7)
    model_b = BinaryMLPScratch(n_features=3, hidden_dim=4, seed=8)

    assert not np.allclose(model_a.W1, model_b.W1)


def test_relu() -> None:
    z = np.array([-2.0, 0.0, 3.0])

    activations = BinaryMLPScratch._relu(z)

    assert np.array_equal(activations, np.array([0.0, 0.0, 3.0]))


def test_stable_sigmoid() -> None:
    z = np.array([0.0, 1000.0, -1000.0])

    probabilities = BinaryMLPScratch._sigmoid(z)

    assert probabilities[0] == pytest.approx(0.5)
    assert probabilities[1] == pytest.approx(1.0)
    assert probabilities[2] == pytest.approx(0.0)


def test_forward_shapes() -> None:
    model = BinaryMLPScratch(n_features=3, hidden_dim=4, seed=42)
    X = np.ones((5, 3))

    probabilities, cache = model.forward(X)

    assert probabilities.shape == (5,)
    assert cache["Z1"].shape == (5, 4)
    assert cache["A1"].shape == (5, 4)
    assert cache["Z2"].shape == (5, 1)


def test_forward_exact_values_with_manual_parameters() -> None:
    model = BinaryMLPScratch(n_features=2, hidden_dim=2, seed=42)
    model.W1 = np.array([[1.0, -1.0], [0.5, 2.0]])
    model.b1 = np.array([0.0, 0.0])
    model.W2 = np.array([[1.0], [-1.0]])
    model.b2 = np.array([0.0])
    X = np.array([[1.0, 1.0], [-1.0, 1.0]])

    probabilities, cache = model.forward(X)

    expected_Z1 = np.array([[1.5, 1.0], [-0.5, 3.0]])
    expected_A1 = np.array([[1.5, 1.0], [0.0, 3.0]])
    expected_Z2 = np.array([[0.5], [-3.0]])
    expected_probabilities = np.array(
        [
            1.0 / (1.0 + np.exp(-0.5)),
            1.0 / (1.0 + np.exp(3.0)),
        ]
    )

    assert np.allclose(cache["X"], X)
    assert np.allclose(cache["Z1"], expected_Z1)
    assert np.allclose(cache["A1"], expected_A1)
    assert np.allclose(cache["Z2"], expected_Z2)
    assert np.allclose(probabilities, expected_probabilities)


def test_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        BinaryMLPScratch(n_features=0, hidden_dim=4)

    with pytest.raises(ValueError):
        BinaryMLPScratch(n_features=3, hidden_dim=0)


def test_invalid_X_shape() -> None:
    model = BinaryMLPScratch(n_features=3, hidden_dim=4)

    with pytest.raises(ValueError):
        model.forward(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        model.forward(np.ones((5, 2)))
