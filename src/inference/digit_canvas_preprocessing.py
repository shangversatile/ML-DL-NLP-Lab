"""Preprocessing utilities for local handwritten digit canvas inputs."""

from numbers import Integral, Real

import numpy as np


def validate_grayscale_image(
    image: np.ndarray,
) -> None:
    """
    Validate a 2D grayscale image array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional.")
    if image.size == 0:
        raise ValueError("image must not be empty.")

    try:
        is_finite = np.all(np.isfinite(image))
    except TypeError as error:
        raise ValueError("image values must be finite numeric values.") from error
    if not is_finite:
        raise ValueError("image values must be finite.")


def normalize_to_unit_interval(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize a grayscale image to the [0, 1] interval.
    """
    validate_grayscale_image(image)

    normalized = np.array(image, dtype=float, copy=True)
    min_value = float(np.min(normalized))
    max_value = float(np.max(normalized))

    if max_value > 1.0 or min_value < 0.0:
        if min_value < 0.0 or max_value > 255.0:
            raise ValueError("image values must be in [0, 1] or [0, 255].")
        normalized = normalized / 255.0

    return normalized


def ensure_bright_foreground(
    image: np.ndarray,
) -> np.ndarray:
    """
    Ensure digit strokes are bright and background is dark.
    """
    normalized = normalize_to_unit_interval(image)

    border_pixels = np.concatenate(
        [
            normalized[0, :],
            normalized[-1, :],
            normalized[:, 0],
            normalized[:, -1],
        ]
    )
    border_mean = float(np.mean(border_pixels))

    if border_mean > 0.5:
        normalized = 1.0 - normalized

    return normalized


def _validate_normalized_image(
    image: np.ndarray,
) -> None:
    validate_grayscale_image(image)
    if np.min(image) < 0.0 or np.max(image) > 1.0:
        raise ValueError("image must be normalized to [0, 1].")


def _validate_threshold(
    threshold: float,
) -> None:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("threshold must be numeric and not boolean.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")


def _validate_unit_real(
    name: str,
    value: float,
) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric and not boolean.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")


def find_foreground_bounding_box(
    image: np.ndarray,
    threshold: float = 0.05,
) -> tuple[int, int, int, int] | None:
    """
    Return bounding box around foreground pixels.

    Returns:
        (row_min, row_max, col_min, col_max), inclusive bounds,
        or None if no foreground exists.
    """
    _validate_normalized_image(image)
    _validate_threshold(threshold)

    foreground_mask = image > threshold
    if not np.any(foreground_mask):
        return None

    rows, columns = np.nonzero(foreground_mask)
    return (
        int(np.min(rows)),
        int(np.max(rows)),
        int(np.min(columns)),
        int(np.max(columns)),
    )


def _validate_padding(
    padding: int,
) -> None:
    if isinstance(padding, (bool, np.bool_)) or not isinstance(padding, Integral):
        raise TypeError("padding must be an integer.")
    if padding < 0:
        raise ValueError("padding must be non-negative.")


def crop_to_foreground(
    image: np.ndarray,
    padding: int = 4,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Crop image around foreground pixels with optional padding.
    """
    _validate_padding(padding)
    normalized = normalize_to_unit_interval(image)
    bounding_box = find_foreground_bounding_box(normalized, threshold=threshold)

    if bounding_box is None:
        return normalized.copy()

    row_min, row_max, col_min, col_max = bounding_box
    row_start = max(0, row_min - padding)
    row_end = min(normalized.shape[0], row_max + padding + 1)
    col_start = max(0, col_min - padding)
    col_end = min(normalized.shape[1], col_max + padding + 1)

    return normalized[row_start:row_end, col_start:col_end].copy()


def resize_to_8x8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Resize a 2D image to 8x8 using simple area-style averaging.
    """
    normalized = normalize_to_unit_interval(image)
    height, width = normalized.shape
    resized = np.zeros((8, 8), dtype=float)

    for output_row in range(8):
        row_start = int(np.floor(output_row * height / 8))
        row_end = int(np.floor((output_row + 1) * height / 8))
        row_start = min(row_start, height - 1)
        row_end = min(height, max(row_start + 1, row_end))

        for output_col in range(8):
            col_start = int(np.floor(output_col * width / 8))
            col_end = int(np.floor((output_col + 1) * width / 8))
            col_start = min(col_start, width - 1)
            col_end = min(width, max(col_start + 1, col_end))

            patch = normalized[row_start:row_end, col_start:col_end]
            resized[output_row, output_col] = float(np.mean(patch))

    return resized


def preprocess_canvas_image(
    image: np.ndarray,
    padding: int = 4,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Convert a canvas grayscale image into a flattened 64-dimensional digit feature vector.
    """
    stages = preprocess_canvas_image_with_stages(
        image,
        padding=padding,
        threshold=threshold,
    )
    return np.asarray(stages["feature_vector"], dtype=float).copy()


def preprocess_canvas_image_with_stages(
    image: np.ndarray,
    padding: int = 4,
    threshold: float = 0.05,
    blank_mass_threshold: float = 0.01,
) -> dict[str, np.ndarray | tuple[int, int, int, int] | None | bool | float]:
    """
    Return all intermediate preprocessing stages for debugging real canvas inputs.
    """
    _validate_padding(padding)
    _validate_threshold(threshold)
    _validate_unit_real("blank_mass_threshold", blank_mass_threshold)

    normalized = normalize_to_unit_interval(image)
    bright_foreground = ensure_bright_foreground(normalized)
    bounding_box = find_foreground_bounding_box(
        bright_foreground,
        threshold=threshold,
    )
    cropped = crop_to_foreground(
        bright_foreground,
        padding=padding,
        threshold=threshold,
    )
    resized = resize_to_8x8(cropped)
    clipped = np.clip(resized, 0.0, 1.0)
    feature_vector = clipped.reshape(64)
    foreground_mass = float(np.mean(bright_foreground))
    is_blank = bool(foreground_mass < blank_mass_threshold)

    return {
        "normalized": normalized.copy(),
        "bright_foreground": bright_foreground.copy(),
        "bounding_box": bounding_box,
        "cropped": cropped.copy(),
        "resized_8x8": clipped.copy(),
        "feature_vector": feature_vector.copy(),
        "is_blank": is_blank,
        "foreground_mass": foreground_mass,
    }
