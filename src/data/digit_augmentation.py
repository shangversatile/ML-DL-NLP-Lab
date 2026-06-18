"""Training-time augmentation utilities for 8x8 handwritten digits."""

from numbers import Integral

import numpy as np

from src.evaluation.shift_diagnostics import (
    apply_shift_condition,
    flatten_digit_images,
    validate_digit_image_batch,
)


def validate_augmentation_conditions(
    conditions: list[dict],
) -> None:
    """
    Validate augmentation condition dictionaries.
    """
    if not isinstance(conditions, list):
        raise TypeError("conditions must be a list.")
    if len(conditions) == 0:
        raise ValueError("conditions must not be empty.")

    seen_names = set()
    tiny_batch = np.zeros((1, 8, 8), dtype=float)
    for condition in conditions:
        if not isinstance(condition, dict):
            raise TypeError("each condition must be a dictionary.")
        if "name" not in condition:
            raise ValueError("each condition must contain 'name'.")
        if "type" not in condition:
            raise ValueError("each condition must contain 'type'.")
        if not isinstance(condition["name"], str):
            raise TypeError("condition names must be strings.")
        if condition["name"] in seen_names:
            raise ValueError("condition names must be unique.")

        apply_shift_condition(tiny_batch, condition, seed=0)
        seen_names.add(condition["name"])


def _validate_labels(
    y: np.ndarray,
    n_samples: int,
) -> None:
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a NumPy array.")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if y.shape[0] != n_samples:
        raise ValueError("y must have one label per image.")
    if np.issubdtype(y.dtype, np.bool_) or not np.issubdtype(
        y.dtype,
        np.integer,
    ):
        raise ValueError("y must contain integer labels.")


def _validate_optional_seed(
    seed: int | None,
) -> None:
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
    ):
        raise TypeError("seed must be an integer or None.")


def create_augmented_digit_dataset(
    images: np.ndarray,
    y: np.ndarray,
    conditions: list[dict],
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Apply augmentation conditions to a digit image batch and return stacked features and labels.
    """
    validate_digit_image_batch(images)
    _validate_labels(y, images.shape[0])
    validate_augmentation_conditions(conditions)
    _validate_optional_seed(seed)

    augmented_images = []
    augmented_features = []
    augmented_labels = []
    augmented_condition_names = []

    for condition_index, condition in enumerate(conditions):
        condition_seed = None if seed is None else int(seed) + condition_index
        transformed_images = apply_shift_condition(
            images,
            condition,
            seed=condition_seed,
        )
        validate_digit_image_batch(transformed_images)

        augmented_images.append(transformed_images.copy())
        augmented_features.append(flatten_digit_images(transformed_images))
        augmented_labels.append(y.copy())
        augmented_condition_names.append(
            np.full(
                y.shape[0],
                condition["name"],
                dtype=object,
            )
        )

    images_augmented = np.vstack(augmented_images)
    X_augmented = np.vstack(augmented_features)
    y_augmented = np.concatenate(augmented_labels).astype(y.dtype, copy=False)
    condition_names = np.concatenate(augmented_condition_names)

    if np.min(images_augmented) < 0.0 or np.max(images_augmented) > 1.0:
        raise ValueError("augmented images must contain values in [0, 1].")

    return {
        "X": X_augmented,
        "y": y_augmented.copy(),
        "images": images_augmented.copy(),
        "condition_names": condition_names.copy(),
    }
