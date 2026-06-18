import numpy as np
import pytest

from src.inference.digits_inference import (
    prepare_digit_features,
    predict_digit_batch,
    predict_single_digit,
)
from src.models.multiclass_mlp import MulticlassMLPScratch


def _create_model() -> MulticlassMLPScratch:
    return MulticlassMLPScratch(
        n_features=64,
        hidden_dim=8,
        num_classes=3,
        seed=42,
    )


def test_prepare_digit_features_accepts_single_flattened_digit() -> None:
    features = prepare_digit_features(np.arange(64))

    assert features.shape == (1, 64)


def test_prepare_digit_features_accepts_single_image() -> None:
    features = prepare_digit_features(np.arange(64).reshape(8, 8))

    assert features.shape == (1, 64)


def test_prepare_digit_features_accepts_batch_of_flattened_digits() -> None:
    inputs = np.arange(128).reshape(2, 64)

    features = prepare_digit_features(inputs)

    assert features.shape == (2, 64)
    np.testing.assert_array_equal(features, inputs.astype(float))


def test_prepare_digit_features_accepts_batch_of_images() -> None:
    inputs = np.arange(128).reshape(2, 8, 8)

    features = prepare_digit_features(inputs)

    assert features.shape == (2, 64)


def test_prepare_digit_features_returns_float_copy() -> None:
    inputs = np.arange(64)

    features = prepare_digit_features(inputs)
    features[0, 0] = -999.0

    assert features.dtype.kind == "f"
    assert inputs[0] == 0


@pytest.mark.parametrize(
    "inputs",
    [
        np.zeros((7, 8)),
        np.zeros((2, 8, 7)),
        np.zeros((2, 65)),
        np.zeros((2, 4, 4, 4)),
    ],
)
def test_prepare_digit_features_rejects_invalid_shapes(inputs: np.ndarray) -> None:
    with pytest.raises(ValueError):
        prepare_digit_features(inputs)


def test_prepare_digit_features_rejects_non_array() -> None:
    with pytest.raises(TypeError):
        prepare_digit_features([0.0] * 64)


def test_prepare_digit_features_rejects_non_finite_values() -> None:
    inputs = np.zeros(64)
    inputs[0] = np.nan

    with pytest.raises(ValueError):
        prepare_digit_features(inputs)


def test_predict_digit_batch_returns_expected_shapes_and_keys() -> None:
    model = _create_model()
    inputs = np.linspace(0.0, 1.0, 128).reshape(2, 64)

    results = predict_digit_batch(model, inputs, top_k=2)

    assert set(results) == {
        "probabilities",
        "predictions",
        "confidences",
        "top_k_indices",
        "top_k_probabilities",
    }
    assert results["probabilities"].shape == (2, 3)
    assert results["predictions"].shape == (2,)
    assert results["confidences"].shape == (2,)
    assert results["top_k_indices"].shape == (2, 2)
    assert results["top_k_probabilities"].shape == (2, 2)
    assert np.all(np.diff(results["top_k_probabilities"], axis=1) <= 0.0)


@pytest.mark.parametrize(
    "top_k,expected_error",
    [
        (0, ValueError),
        (4, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_predict_digit_batch_rejects_invalid_top_k(
    top_k: object,
    expected_error: type[Exception],
) -> None:
    model = _create_model()
    inputs = np.zeros((2, 64))

    with pytest.raises(expected_error):
        predict_digit_batch(model, inputs, top_k=top_k)


def test_predict_single_digit_accepts_flattened_input() -> None:
    model = _create_model()

    result = predict_single_digit(model, np.zeros(64), top_k=2)

    assert result["probabilities"].shape == (3,)
    assert isinstance(result["prediction"], int)
    assert isinstance(result["confidence"], float)
    assert result["top_k_indices"].shape == (2,)
    assert result["top_k_probabilities"].shape == (2,)


def test_predict_single_digit_accepts_image_input() -> None:
    model = _create_model()

    result = predict_single_digit(model, np.zeros((8, 8)), top_k=2)

    assert result["probabilities"].shape == (3,)
    assert isinstance(result["prediction"], int)
    assert isinstance(result["confidence"], float)
    assert result["top_k_indices"].shape == (2,)
    assert result["top_k_probabilities"].shape == (2,)


def test_predict_single_digit_rejects_multi_sample_input() -> None:
    model = _create_model()

    with pytest.raises(ValueError):
        predict_single_digit(model, np.zeros((2, 64)), top_k=2)
