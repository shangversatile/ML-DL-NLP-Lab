"""Distribution-shift probes for 8x8 handwritten-digit inputs."""

from numbers import Integral, Real

import numpy as np


def validate_digit_image_batch(
    images: np.ndarray,
) -> None:
    """
    Validate a batch of 8x8 digit images.
    """
    if not isinstance(images, np.ndarray):
        raise TypeError("images must be a NumPy array.")
    if images.ndim != 3 or images.shape[1:] != (8, 8):
        raise ValueError("images must have shape (n_samples, 8, 8).")
    if images.shape[0] == 0:
        raise ValueError("images must contain at least one sample.")
    if np.issubdtype(images.dtype, np.bool_) or not np.issubdtype(
        images.dtype,
        np.number,
    ):
        raise ValueError("images must contain numeric values.")
    if not np.all(np.isfinite(images)):
        raise ValueError("images must contain only finite values.")
    if np.min(images) < 0.0 or np.max(images) > 1.0:
        raise ValueError("images must contain values in [0, 1].")


def _validate_integer_shift(
    name: str,
    value: int,
) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")


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


def _validate_non_negative_real(
    name: str,
    value: float,
) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric and not boolean.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def identity_images(
    images: np.ndarray,
) -> np.ndarray:
    """
    Return a copy of the input image batch.
    """
    validate_digit_image_batch(images)
    return images.copy()


def shift_images(
    images: np.ndarray,
    row_shift: int,
    col_shift: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Shift 8x8 images with zero-like padding.
    """
    validate_digit_image_batch(images)
    _validate_integer_shift("row_shift", row_shift)
    _validate_integer_shift("col_shift", col_shift)
    _validate_unit_real("fill_value", fill_value)

    shifted = np.full_like(images, fill_value, dtype=float)

    source_row_start = max(0, -row_shift)
    source_row_end = min(8, 8 - row_shift)
    target_row_start = max(0, row_shift)
    target_row_end = min(8, 8 + row_shift)

    source_col_start = max(0, -col_shift)
    source_col_end = min(8, 8 - col_shift)
    target_col_start = max(0, col_shift)
    target_col_end = min(8, 8 + col_shift)

    if source_row_end > source_row_start and source_col_end > source_col_start:
        shifted[
            :,
            target_row_start:target_row_end,
            target_col_start:target_col_end,
        ] = images[
            :,
            source_row_start:source_row_end,
            source_col_start:source_col_end,
        ]

    return shifted


def scale_intensity(
    images: np.ndarray,
    factor: float,
) -> np.ndarray:
    """
    Multiply pixel intensities and clip to [0, 1].
    """
    validate_digit_image_batch(images)
    _validate_non_negative_real("factor", factor)
    return np.clip(images * factor, 0.0, 1.0)


def add_pixel_noise(
    images: np.ndarray,
    noise_std: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add Gaussian pixel noise and clip to [0, 1].
    """
    validate_digit_image_batch(images)
    _validate_non_negative_real("noise_std", noise_std)

    if noise_std == 0:
        return images.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=images.shape)
    return np.clip(images + noise, 0.0, 1.0)


def threshold_images(
    images: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Binarize images at a threshold.
    """
    validate_digit_image_batch(images)
    _validate_unit_real("threshold", threshold)
    return (images >= threshold).astype(float)


def thicken_strokes(
    images: np.ndarray,
) -> np.ndarray:
    """
    Thicken bright strokes using a simple 3x3 max filter.
    """
    validate_digit_image_batch(images)
    padded = np.pad(
        images,
        pad_width=((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=0.0,
    )
    thickened = np.zeros_like(images, dtype=float)

    for row in range(8):
        for col in range(8):
            neighborhood = padded[:, row : row + 3, col : col + 3]
            thickened[:, row, col] = np.max(neighborhood, axis=(1, 2))

    return thickened


def thin_strokes(
    images: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Produce a crude thinning probe by keeping only high-intensity pixels.
    """
    validate_digit_image_batch(images)
    _validate_unit_real("threshold", threshold)
    return np.where(images >= threshold, images, 0.0)


def flatten_digit_images(
    images: np.ndarray,
) -> np.ndarray:
    """
    Flatten an image batch into shape (n_samples, 64).
    """
    validate_digit_image_batch(images)
    return images.copy().reshape(images.shape[0], 64)


def apply_shift_condition(
    images: np.ndarray,
    condition: dict,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply a shift/probe condition dictionary to digit images.
    """
    validate_digit_image_batch(images)
    if not isinstance(condition, dict):
        raise TypeError("condition must be a dictionary.")
    if "name" not in condition:
        raise ValueError("condition must contain 'name'.")
    if "type" not in condition:
        raise ValueError("condition must contain 'type'.")

    condition_type = condition["type"]
    if condition_type == "identity":
        return identity_images(images)
    if condition_type == "shift":
        return shift_images(
            images,
            row_shift=condition["row_shift"],
            col_shift=condition["col_shift"],
            fill_value=condition.get("fill_value", 0.0),
        )
    if condition_type == "intensity":
        return scale_intensity(images, factor=condition["factor"])
    if condition_type == "noise":
        return add_pixel_noise(
            images,
            noise_std=condition["noise_std"],
            seed=seed,
        )
    if condition_type == "threshold":
        return threshold_images(images, threshold=condition["threshold"])
    if condition_type == "thicken":
        return thicken_strokes(images)
    if condition_type == "thin":
        return thin_strokes(
            images,
            threshold=condition.get("threshold", 0.5),
        )

    raise ValueError(f"unsupported shift condition type: {condition_type!r}")
