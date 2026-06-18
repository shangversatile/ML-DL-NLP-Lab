"""Tests for saving and loading user-drawn canvas samples."""

import numpy as np
import pytest

from src.inference.canvas_sample_store import (
    load_canvas_samples,
    save_canvas_sample,
    stack_canvas_sample_features,
    validate_optional_digit_label,
)
from src.inference.digit_canvas_preprocessing import preprocess_canvas_image_with_stages


def _raw_canvas() -> np.ndarray:
    canvas = np.full((20, 20), 255.0)
    canvas[7:13, 8:12] = 0.0
    return canvas


def _stages() -> dict:
    return preprocess_canvas_image_with_stages(_raw_canvas())


def _prediction() -> dict:
    probabilities = np.array([0.05, 0.80, 0.15])
    return {
        "probabilities": probabilities,
        "prediction": 1,
        "confidence": 0.80,
        "top_k_indices": np.array([1, 2, 0]),
        "top_k_probabilities": np.array([0.80, 0.15, 0.05]),
    }


def test_validate_optional_digit_label_accepts_none_and_digits() -> None:
    validate_optional_digit_label(None)
    for label in range(10):
        validate_optional_digit_label(label)


@pytest.mark.parametrize("label", [True, -1, 10, 1.5, "1"])
def test_validate_optional_digit_label_rejects_invalid_labels(label) -> None:
    expected_error = TypeError if isinstance(label, (bool, float, str)) else ValueError
    with pytest.raises(expected_error):
        validate_optional_digit_label(label)


def test_save_canvas_sample_creates_npz_with_expected_shapes(tmp_path) -> None:
    sample_path = save_canvas_sample(
        tmp_path,
        _raw_canvas(),
        _stages(),
        _prediction(),
        checkpoint_path="checkpoint.npz",
        true_label=1,
    )

    assert sample_path.exists()
    assert sample_path.suffix == ".npz"
    with np.load(sample_path, allow_pickle=False) as data:
        assert data["raw_canvas"].shape == (20, 20)
        assert data["resized_8x8"].shape == (8, 8)
        assert data["feature_vector"].shape == (64,)
        assert data["probabilities"].shape == (3,)
        assert int(data["true_label"]) == 1


def test_load_canvas_samples_missing_directory_returns_empty_list(tmp_path) -> None:
    assert load_canvas_samples(tmp_path / "missing") == []


def test_load_canvas_samples_filters_labeled_and_unlabeled_samples(tmp_path) -> None:
    save_canvas_sample(
        tmp_path,
        _raw_canvas(),
        _stages(),
        _prediction(),
        checkpoint_path="checkpoint.npz",
        true_label=1,
    )
    save_canvas_sample(
        tmp_path,
        _raw_canvas(),
        _stages(),
        _prediction(),
        checkpoint_path="checkpoint.npz",
        true_label=None,
    )

    labeled = load_canvas_samples(tmp_path, require_labels=True)
    all_samples = load_canvas_samples(tmp_path, require_labels=False)

    assert len(labeled) == 1
    assert labeled[0]["true_label"] == 1
    assert len(all_samples) == 2
    assert {sample["true_label"] for sample in all_samples} == {-1, 1}


def test_stack_canvas_sample_features_returns_arrays(tmp_path) -> None:
    save_canvas_sample(
        tmp_path,
        _raw_canvas(),
        _stages(),
        _prediction(),
        checkpoint_path="checkpoint.npz",
        true_label=1,
    )
    samples = load_canvas_samples(tmp_path, require_labels=True)

    X, y = stack_canvas_sample_features(samples)

    assert X.shape == (1, 64)
    assert y.shape == (1,)
    assert y[0] == 1


def test_stack_canvas_sample_features_rejects_empty_or_unlabeled_samples(tmp_path) -> None:
    with pytest.raises(ValueError):
        stack_canvas_sample_features([])

    save_canvas_sample(
        tmp_path,
        _raw_canvas(),
        _stages(),
        _prediction(),
        checkpoint_path="checkpoint.npz",
        true_label=None,
    )
    samples = load_canvas_samples(tmp_path, require_labels=False)

    with pytest.raises(ValueError):
        stack_canvas_sample_features(samples)
