"""Tests for training-time digit augmentation utilities."""

import numpy as np
import pytest

from src.data.digit_augmentation import (
    create_augmented_digit_dataset,
    validate_augmentation_conditions,
)


def _sample_images() -> np.ndarray:
    images = np.zeros((2, 8, 8), dtype=float)
    images[0, 3, 3] = 1.0
    images[1, 4, 4] = 0.7
    return images


def test_validate_augmentation_conditions_accepts_valid_conditions() -> None:
    conditions = [
        {"name": "clean", "type": "identity"},
        {
            "name": "shift",
            "type": "shift",
            "row_shift": 1,
            "col_shift": 0,
        },
    ]

    validate_augmentation_conditions(conditions)


def test_validate_augmentation_conditions_rejects_invalid_conditions() -> None:
    with pytest.raises(TypeError):
        validate_augmentation_conditions("clean")

    with pytest.raises(ValueError):
        validate_augmentation_conditions([])

    with pytest.raises(ValueError):
        validate_augmentation_conditions(
            [
                {"name": "clean", "type": "identity"},
                {"name": "clean", "type": "identity"},
            ]
        )

    with pytest.raises(ValueError):
        validate_augmentation_conditions([{"type": "identity"}])

    with pytest.raises(ValueError):
        validate_augmentation_conditions([{"name": "clean"}])

    with pytest.raises(ValueError):
        validate_augmentation_conditions([{"name": "bad", "type": "unknown"}])


def test_create_augmented_digit_dataset_shapes() -> None:
    images = _sample_images()
    y = np.array([1, 2])
    conditions = [
        {"name": "clean", "type": "identity"},
        {
            "name": "shift",
            "type": "shift",
            "row_shift": 1,
            "col_shift": 0,
        },
    ]

    augmented = create_augmented_digit_dataset(images, y, conditions, seed=42)

    assert augmented["X"].shape == (4, 64)
    assert augmented["y"].shape == (4,)
    assert augmented["images"].shape == (4, 8, 8)
    assert augmented["condition_names"].shape == (4,)


def test_create_augmented_digit_dataset_repeats_labels_by_condition() -> None:
    augmented = create_augmented_digit_dataset(
        _sample_images(),
        np.array([1, 2]),
        [
            {"name": "clean", "type": "identity"},
            {"name": "thin", "type": "thin", "threshold": 0.5},
        ],
    )

    np.testing.assert_array_equal(augmented["y"], np.array([1, 2, 1, 2]))


def test_create_augmented_digit_dataset_repeats_condition_names() -> None:
    augmented = create_augmented_digit_dataset(
        _sample_images(),
        np.array([1, 2]),
        [
            {"name": "clean", "type": "identity"},
            {"name": "thin", "type": "thin", "threshold": 0.5},
        ],
    )

    np.testing.assert_array_equal(
        augmented["condition_names"],
        np.array(["clean", "clean", "thin", "thin"], dtype=object),
    )


def test_create_augmented_digit_dataset_values_remain_unit_interval() -> None:
    augmented = create_augmented_digit_dataset(
        _sample_images(),
        np.array([1, 2]),
        [
            {"name": "bright", "type": "intensity", "factor": 2.0},
            {"name": "noise", "type": "noise", "noise_std": 0.1},
        ],
        seed=42,
    )

    assert augmented["X"].min() >= 0.0
    assert augmented["X"].max() <= 1.0
    assert augmented["images"].min() >= 0.0
    assert augmented["images"].max() <= 1.0


def test_create_augmented_digit_dataset_noise_is_reproducible() -> None:
    images = _sample_images()
    y = np.array([1, 2])
    conditions = [{"name": "noise", "type": "noise", "noise_std": 0.1}]

    first = create_augmented_digit_dataset(images, y, conditions, seed=7)
    second = create_augmented_digit_dataset(images, y, conditions, seed=7)

    np.testing.assert_allclose(first["X"], second["X"])
    np.testing.assert_allclose(first["images"], second["images"])


def test_create_augmented_digit_dataset_does_not_mutate_inputs() -> None:
    images = _sample_images()
    y = np.array([1, 2])
    images_before = images.copy()
    y_before = y.copy()

    create_augmented_digit_dataset(
        images,
        y,
        [{"name": "thicken", "type": "thicken"}],
    )

    np.testing.assert_allclose(images, images_before)
    np.testing.assert_array_equal(y, y_before)


def test_create_augmented_digit_dataset_rejects_invalid_labels() -> None:
    images = _sample_images()
    conditions = [{"name": "clean", "type": "identity"}]

    with pytest.raises(TypeError):
        create_augmented_digit_dataset(images, [1, 2], conditions)

    with pytest.raises(ValueError):
        create_augmented_digit_dataset(images, np.array([[1, 2]]), conditions)

    with pytest.raises(ValueError):
        create_augmented_digit_dataset(images, np.array([1]), conditions)

    with pytest.raises(ValueError):
        create_augmented_digit_dataset(images, np.array([1.0, 2.0]), conditions)
