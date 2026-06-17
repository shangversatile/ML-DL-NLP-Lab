"""Tests for the NumPy multiclass MLP."""

import numpy as np
import pytest

from src.models.multiclass_mlp import MulticlassMLPScratch
from src.utils.multiclass import stable_softmax


def create_manual_model() -> MulticlassMLPScratch:
    model = MulticlassMLPScratch(
        n_features=2,
        hidden_dim=2,
        num_classes=3,
        seed=0,
    )

    model.W1 = np.array(
        [
            [1.0, -1.0],
            [0.5, 2.0],
        ]
    )
    model.b1 = np.array([0.0, 0.0])
    model.W2 = np.array(
        [
            [1.0, -1.0, 0.5],
            [0.0, 0.5, -0.5],
        ]
    )
    model.b2 = np.array([0.1, -0.2, 0.3])
    return model


def create_manual_X() -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )


def test_initialization_parameter_shapes() -> None:
    model = MulticlassMLPScratch(n_features=3, hidden_dim=4, num_classes=5)

    assert model.W1.shape == (3, 4)
    assert model.b1.shape == (4,)
    assert model.W2.shape == (4, 5)
    assert model.b2.shape == (5,)


def test_reproducible_initialization_with_same_seed() -> None:
    model_a = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=7,
    )
    model_b = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=7,
    )

    np.testing.assert_allclose(model_a.W1, model_b.W1)
    np.testing.assert_allclose(model_a.W2, model_b.W2)
    np.testing.assert_array_equal(model_a.b1, model_b.b1)
    np.testing.assert_array_equal(model_a.b2, model_b.b2)


def test_different_seeds_initialize_different_weights() -> None:
    model_a = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=7,
    )
    model_b = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=8,
    )

    assert not np.allclose(model_a.W1, model_b.W1)
    assert not np.allclose(model_a.W2, model_b.W2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_features": 0, "hidden_dim": 3, "num_classes": 3},
        {"n_features": -1, "hidden_dim": 3, "num_classes": 3},
        {"n_features": 2, "hidden_dim": 0, "num_classes": 3},
        {"n_features": 2, "hidden_dim": -1, "num_classes": 3},
        {"n_features": 2, "hidden_dim": 3, "num_classes": 0},
        {"n_features": 2, "hidden_dim": 3, "num_classes": 1},
    ],
)
def test_invalid_constructor_value_arguments_raise_value_error(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        MulticlassMLPScratch(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_features": True, "hidden_dim": 3, "num_classes": 3},
        {"n_features": 2, "hidden_dim": False, "num_classes": 3},
        {"n_features": 2, "hidden_dim": 3, "num_classes": True},
        {"n_features": 2.0, "hidden_dim": 3, "num_classes": 3},
        {"n_features": 2, "hidden_dim": "3", "num_classes": 3},
        {"n_features": 2, "hidden_dim": 3, "num_classes": 3.0},
    ],
)
def test_invalid_constructor_type_arguments_raise_type_error(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError):
        MulticlassMLPScratch(**kwargs)


def test_forward_shapes_probability_rows_and_cache() -> None:
    model = MulticlassMLPScratch(
        n_features=2,
        hidden_dim=3,
        num_classes=3,
        seed=42,
    )
    X = np.array(
        [
            [1.0, 2.0],
            [-1.0, 0.5],
        ]
    )

    probabilities, cache = model.forward(X)

    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))
    assert set(cache) == {"X", "Z1", "A1", "logits"}
    assert cache["X"].shape == (2, 2)
    assert cache["Z1"].shape == (2, 3)
    assert cache["A1"].shape == (2, 3)
    assert cache["logits"].shape == (2, 3)


def test_manual_forward_values() -> None:
    model = create_manual_model()
    X = create_manual_X()

    probabilities, cache = model.forward(X)

    expected_Z1 = np.array(
        [
            [1.5, 1.0],
            [-0.5, 3.0],
        ]
    )
    expected_A1 = np.array(
        [
            [1.5, 1.0],
            [0.0, 3.0],
        ]
    )
    expected_logits = np.array(
        [
            [1.6, -1.2, 0.55],
            [0.1, 1.3, -1.2],
        ]
    )

    np.testing.assert_allclose(cache["X"], X)
    np.testing.assert_allclose(cache["Z1"], expected_Z1)
    np.testing.assert_allclose(cache["A1"], expected_A1)
    np.testing.assert_allclose(cache["logits"], expected_logits)
    np.testing.assert_allclose(probabilities, stable_softmax(expected_logits))


def test_predict_proba_matches_forward_probabilities() -> None:
    model = create_manual_model()
    X = create_manual_X()

    forward_probabilities, _ = model.forward(X)
    predicted_probabilities = model.predict_proba(X)

    np.testing.assert_allclose(predicted_probabilities, forward_probabilities)


def test_predict_returns_argmax_integer_labels() -> None:
    model = create_manual_model()
    X = create_manual_X()

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, np.argmax(probabilities, axis=1))
    assert predictions.shape == (2,)
    assert predictions.dtype.kind in {"i", "u"}


def test_compute_loss_matches_manual_cross_entropy() -> None:
    model = create_manual_model()
    X = create_manual_X()
    y = np.array([0, 1])

    probabilities, _ = model.forward(X)
    expected = -np.mean(np.log(probabilities[np.arange(2), y]))

    assert model.compute_loss(X, y) == pytest.approx(expected)


def test_compute_gradients_keys_and_shapes() -> None:
    model = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=42,
    )
    X = np.ones((6, 3))
    y = np.array([0, 1, 2, 3, 4, 0])

    gradients = model.compute_gradients(X, y)

    assert set(gradients) == {"dW1", "db1", "dW2", "db2"}
    assert gradients["dW1"].shape == model.W1.shape
    assert gradients["db1"].shape == model.b1.shape
    assert gradients["dW2"].shape == model.W2.shape
    assert gradients["db2"].shape == model.b2.shape


def test_get_parameters_returns_copies() -> None:
    model = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=42,
    )
    original_value = model.W1[0, 0]

    parameters = model.get_parameters()
    parameters["W1"][0, 0] += 10.0

    assert model.W1[0, 0] == original_value


def test_set_parameters_updates_values_and_copies_inputs() -> None:
    model = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=42,
    )
    parameters = {
        name: parameter + 0.25
        for name, parameter in model.get_parameters().items()
    }

    model.set_parameters(parameters)

    for name in model.PARAMETER_NAMES:
        np.testing.assert_allclose(getattr(model, name), parameters[name])

    model_value = model.W1[0, 0]
    parameters["W1"][0, 0] += 10.0

    assert model.W1[0, 0] == model_value


def test_set_parameters_invalid_keys_raise_value_error() -> None:
    model = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=42,
    )
    parameters = model.get_parameters()
    parameters.pop("W2")

    with pytest.raises(ValueError):
        model.set_parameters(parameters)


def test_set_parameters_invalid_shapes_raise_value_error() -> None:
    model = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=42,
    )
    parameters = model.get_parameters()
    parameters["W1"] = np.ones((2, 2))

    with pytest.raises(ValueError):
        model.set_parameters(parameters)


def test_set_parameters_non_array_values_raise_type_error() -> None:
    model = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=42,
    )
    parameters = model.get_parameters()
    parameters["W1"] = [[1.0]]

    with pytest.raises(TypeError):
        model.set_parameters(parameters)


def test_set_parameters_non_finite_values_raise_value_error() -> None:
    model = MulticlassMLPScratch(
        n_features=3,
        hidden_dim=4,
        num_classes=5,
        seed=42,
    )
    parameters = model.get_parameters()
    parameters["W1"][0, 0] = np.nan

    with pytest.raises(ValueError):
        model.set_parameters(parameters)


def test_non_array_X_raises_type_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)

    with pytest.raises(TypeError):
        model.forward([[1.0, 2.0]])


def test_non_2d_X_raises_value_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)

    with pytest.raises(ValueError):
        model.forward(np.array([1.0, 2.0]))


def test_wrong_feature_dimension_raises_value_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)

    with pytest.raises(ValueError):
        model.forward(np.ones((2, 3)))


def test_non_finite_X_raises_value_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)

    with pytest.raises(ValueError):
        model.forward(np.array([[1.0, np.inf]]))


def test_non_array_y_raises_type_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)
    X = np.ones((2, 2))

    with pytest.raises(TypeError):
        model.compute_loss(X, [0, 1])


def test_non_1d_y_raises_value_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)
    X = np.ones((2, 2))

    with pytest.raises(ValueError):
        model.compute_loss(X, np.array([[0], [1]]))


def test_wrong_y_length_raises_value_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)
    X = np.ones((2, 2))

    with pytest.raises(ValueError):
        model.compute_loss(X, np.array([0]))


def test_non_integer_y_raises_value_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)
    X = np.ones((2, 2))

    with pytest.raises(ValueError):
        model.compute_loss(X, np.array([0.0, 1.0]))


def test_negative_y_raises_value_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)
    X = np.ones((2, 2))

    with pytest.raises(ValueError):
        model.compute_loss(X, np.array([0, -1]))


def test_out_of_range_y_raises_value_error() -> None:
    model = MulticlassMLPScratch(n_features=2, hidden_dim=3, num_classes=3, seed=42)
    X = np.ones((2, 2))

    with pytest.raises(ValueError):
        model.compute_loss(X, np.array([0, 3]))
