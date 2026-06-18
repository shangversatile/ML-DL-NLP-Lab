import numpy as np
import pytest

from src.evaluation.shift_diagnostics import (
    add_pixel_noise,
    apply_shift_condition,
    flatten_digit_images,
    identity_images,
    scale_intensity,
    shift_images,
    thicken_strokes,
    thin_strokes,
    threshold_images,
    validate_digit_image_batch,
)


def _single_pixel_batch() -> np.ndarray:
    images = np.zeros((1, 8, 8))
    images[0, 3, 4] = 1.0
    return images


def test_validate_digit_image_batch_accepts_valid_images() -> None:
    validate_digit_image_batch(np.zeros((2, 8, 8)))


def test_validate_digit_image_batch_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        validate_digit_image_batch([np.zeros((8, 8))])

    with pytest.raises(ValueError):
        validate_digit_image_batch(np.zeros((8, 8)))

    with pytest.raises(ValueError):
        validate_digit_image_batch(np.zeros((0, 8, 8)))

    image = np.zeros((1, 8, 8))
    image[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_digit_image_batch(image)

    with pytest.raises(ValueError):
        validate_digit_image_batch(np.full((1, 8, 8), 1.5))


def test_identity_returns_copy() -> None:
    images = _single_pixel_batch()
    copied = identity_images(images)
    copied[0, 3, 4] = 0.0

    assert images[0, 3, 4] == pytest.approx(1.0)


def test_shift_down_and_right() -> None:
    images = _single_pixel_batch()

    shifted = shift_images(images, row_shift=1, col_shift=2)

    assert shifted[0, 4, 6] == pytest.approx(1.0)
    assert shifted[0, 3, 4] == pytest.approx(0.0)
    assert shifted[0, 0, 0] == pytest.approx(0.0)


def test_shift_up_and_left() -> None:
    images = _single_pixel_batch()

    shifted = shift_images(images, row_shift=-2, col_shift=-3)

    assert shifted[0, 1, 1] == pytest.approx(1.0)
    assert shifted[0, 3, 4] == pytest.approx(0.0)


def test_shift_has_no_wraparound() -> None:
    images = np.zeros((1, 8, 8))
    images[0, 0, 0] = 1.0

    shifted = shift_images(images, row_shift=-1, col_shift=0)

    assert np.sum(shifted) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "row_shift,col_shift,fill_value,expected_error",
    [
        (True, 0, 0.0, TypeError),
        (0, 1.5, 0.0, TypeError),
        (0, 0, True, TypeError),
        (0, 0, -0.1, ValueError),
        (0, 0, 1.1, ValueError),
    ],
)
def test_shift_rejects_invalid_arguments(
    row_shift: object,
    col_shift: object,
    fill_value: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        shift_images(
            _single_pixel_batch(),
            row_shift=row_shift,
            col_shift=col_shift,
            fill_value=fill_value,
        )


def test_intensity_scaling() -> None:
    images = np.full((1, 8, 8), 0.6)

    dimmed = scale_intensity(images, factor=0.5)
    brightened = scale_intensity(images, factor=2.0)

    assert np.allclose(dimmed, 0.3)
    assert np.allclose(brightened, 1.0)


@pytest.mark.parametrize("factor,expected_error", [(-0.1, ValueError), (True, TypeError), ("2", TypeError)])
def test_intensity_scaling_rejects_invalid_factor(
    factor: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        scale_intensity(_single_pixel_batch(), factor=factor)


def test_noise_reproducibility() -> None:
    images = np.full((1, 8, 8), 0.5)

    first = add_pixel_noise(images, noise_std=0.1, seed=42)
    second = add_pixel_noise(images, noise_std=0.1, seed=42)

    np.testing.assert_allclose(first, second)


def test_zero_noise_returns_copy() -> None:
    images = _single_pixel_batch()

    noisy = add_pixel_noise(images, noise_std=0.0, seed=42)
    noisy[0, 3, 4] = 0.0

    assert images[0, 3, 4] == pytest.approx(1.0)


@pytest.mark.parametrize("noise_std,expected_error", [(-0.1, ValueError), (True, TypeError), ("0.1", TypeError)])
def test_noise_rejects_invalid_noise_std(
    noise_std: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        add_pixel_noise(_single_pixel_batch(), noise_std=noise_std)


def test_threshold_images() -> None:
    images = np.array([[[0.2, 0.5], [0.6, 0.1]]], dtype=float)
    padded = np.zeros((1, 8, 8))
    padded[:, :2, :2] = images

    thresholded = threshold_images(padded, threshold=0.5)

    np.testing.assert_array_equal(
        thresholded[0, :2, :2],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )


@pytest.mark.parametrize("threshold,expected_error", [(-0.1, ValueError), (1.1, ValueError), (True, TypeError), ("0.5", TypeError)])
def test_threshold_rejects_invalid_threshold(
    threshold: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        threshold_images(_single_pixel_batch(), threshold=threshold)


def test_thicken_strokes_expands_single_pixel() -> None:
    images = _single_pixel_batch()

    thickened = thicken_strokes(images)

    assert np.all(thickened[0, 2:5, 3:6] == 1.0)
    assert thickened[0, 0, 0] == pytest.approx(0.0)


def test_thin_strokes_removes_low_intensity_and_preserves_high_intensity() -> None:
    images = np.zeros((1, 8, 8))
    images[0, 2, 2] = 0.4
    images[0, 3, 3] = 0.8

    thinned = thin_strokes(images, threshold=0.5)

    assert thinned[0, 2, 2] == pytest.approx(0.0)
    assert thinned[0, 3, 3] == pytest.approx(0.8)


def test_flatten_digit_images() -> None:
    images = np.zeros((3, 8, 8))

    flattened = flatten_digit_images(images)

    assert flattened.shape == (3, 64)


@pytest.mark.parametrize(
    "condition",
    [
        {"name": "clean", "type": "identity"},
        {"name": "shift", "type": "shift", "row_shift": 1, "col_shift": 0},
        {"name": "intensity", "type": "intensity", "factor": 0.5},
        {"name": "noise", "type": "noise", "noise_std": 0.1},
        {"name": "threshold", "type": "threshold", "threshold": 0.5},
        {"name": "thicken", "type": "thicken"},
        {"name": "thin", "type": "thin", "threshold": 0.5},
    ],
)
def test_apply_shift_condition_supported_types(condition: dict) -> None:
    transformed = apply_shift_condition(
        _single_pixel_batch(),
        condition,
        seed=42,
    )

    assert transformed.shape == (1, 8, 8)
    assert np.min(transformed) >= 0.0
    assert np.max(transformed) <= 1.0


def test_apply_shift_condition_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        apply_shift_condition(
            _single_pixel_batch(),
            {"name": "bad", "type": "unknown"},
        )
