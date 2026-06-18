import numpy as np
import pytest

from src.inference.digit_canvas_preprocessing import (
    crop_to_foreground,
    ensure_bright_foreground,
    find_foreground_bounding_box,
    normalize_to_unit_interval,
    preprocess_canvas_image,
    preprocess_canvas_image_with_stages,
    resize_to_8x8,
    validate_grayscale_image,
)


def test_normalize_0_to_255_image() -> None:
    image = np.array([[0, 255]], dtype=float)

    normalized = normalize_to_unit_interval(image)

    np.testing.assert_allclose(normalized, np.array([[0.0, 1.0]]))


def test_already_normalized_image_returns_copy() -> None:
    image = np.array([[0.0, 1.0]], dtype=float)

    normalized = normalize_to_unit_interval(image)
    normalized[0, 0] = 0.5

    assert image[0, 0] == 0.0
    assert normalized[0, 0] == 0.5


def test_invalid_normalization_range() -> None:
    with pytest.raises(ValueError):
        normalize_to_unit_interval(np.array([[-1.0, 0.0]]))

    with pytest.raises(ValueError):
        normalize_to_unit_interval(np.array([[0.0, 256.0]]))


def test_bright_background_inversion() -> None:
    image = np.ones((5, 5))
    image[2, 2] = 0.0

    converted = ensure_bright_foreground(image)

    assert converted[2, 2] == pytest.approx(1.0)
    assert converted[0, 0] == pytest.approx(0.0)


def test_dark_background_image_not_inverted() -> None:
    image = np.zeros((5, 5))
    image[2, 2] = 1.0

    converted = ensure_bright_foreground(image)

    assert converted[2, 2] == pytest.approx(1.0)
    assert converted[0, 0] == pytest.approx(0.0)


def test_bounding_box_detection() -> None:
    image = np.zeros((6, 7))
    image[2:4, 3:6] = 0.8

    bounding_box = find_foreground_bounding_box(image, threshold=0.05)

    assert bounding_box == (2, 3, 3, 5)


def test_no_foreground_returns_none() -> None:
    image = np.zeros((4, 4))

    assert find_foreground_bounding_box(image, threshold=0.05) is None


def test_crop_with_padding() -> None:
    image = np.zeros((10, 10))
    image[4:6, 4:6] = 1.0

    cropped = crop_to_foreground(image, padding=2, threshold=0.05)

    assert cropped.shape == (6, 6)
    assert np.max(cropped) == pytest.approx(1.0)


def test_crop_with_padding_respects_boundaries() -> None:
    image = np.zeros((5, 5))
    image[0, 0] = 1.0

    cropped = crop_to_foreground(image, padding=4, threshold=0.05)

    assert cropped.shape == (5, 5)


def test_resize_to_8x8() -> None:
    image = np.linspace(0.0, 1.0, 400).reshape(20, 20)

    resized = resize_to_8x8(image)

    assert resized.shape == (8, 8)
    assert np.all(np.isfinite(resized))
    assert np.min(resized) >= 0.0
    assert np.max(resized) <= 1.0


def test_full_preprocessing_returns_flat_vector_for_bright_foreground() -> None:
    image = np.zeros((20, 20))
    image[7:13, 8:12] = 1.0

    features = preprocess_canvas_image(image)

    assert features.shape == (64,)
    assert np.min(features) >= 0.0
    assert np.max(features) <= 1.0


def test_full_preprocessing_returns_flat_vector_for_dark_foreground() -> None:
    image = np.full((20, 20), 255.0)
    image[7:13, 8:12] = 0.0

    features = preprocess_canvas_image(image)

    assert features.shape == (64,)
    assert np.min(features) >= 0.0
    assert np.max(features) <= 1.0
    assert np.max(features) > 0.0


def test_preprocess_canvas_image_with_stages_returns_required_keys() -> None:
    image = np.full((20, 20), 255.0)
    image[7:13, 8:12] = 0.0

    stages = preprocess_canvas_image_with_stages(image)

    assert set(stages) == {
        "normalized",
        "bright_foreground",
        "bounding_box",
        "cropped",
        "resized_8x8",
        "feature_vector",
        "is_blank",
        "foreground_mass",
    }
    assert stages["resized_8x8"].shape == (8, 8)
    assert stages["feature_vector"].shape == (64,)
    assert np.isfinite(stages["foreground_mass"])
    assert stages["foreground_mass"] >= 0.0


def test_stage_aware_preprocessing_detects_blank_image() -> None:
    image = np.full((20, 20), 255.0)

    stages = preprocess_canvas_image_with_stages(image)

    assert stages["is_blank"] is True
    assert stages["bounding_box"] is None


def test_stage_aware_preprocessing_detects_non_blank_image() -> None:
    image = np.full((20, 20), 255.0)
    image[7:13, 8:12] = 0.0

    stages = preprocess_canvas_image_with_stages(image)

    assert stages["is_blank"] is False
    assert stages["bounding_box"] is not None


def test_preprocess_canvas_image_matches_stage_feature_vector() -> None:
    image = np.full((20, 20), 255.0)
    image[7:13, 8:12] = 0.0

    features = preprocess_canvas_image(image)
    stages = preprocess_canvas_image_with_stages(image)

    np.testing.assert_allclose(features, stages["feature_vector"])


@pytest.mark.parametrize(
    "blank_mass_threshold,expected_error",
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (True, TypeError),
        ("0.01", TypeError),
    ],
)
def test_stage_aware_preprocessing_rejects_invalid_blank_mass_threshold(
    blank_mass_threshold: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        preprocess_canvas_image_with_stages(
            np.zeros((4, 4)),
            blank_mass_threshold=blank_mass_threshold,
        )


def test_validate_grayscale_image_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        validate_grayscale_image([[0.0]])

    with pytest.raises(ValueError):
        validate_grayscale_image(np.zeros((2, 2, 1)))

    with pytest.raises(ValueError):
        validate_grayscale_image(np.array([]).reshape(0, 0))

    image = np.zeros((2, 2))
    image[0, 0] = np.inf
    with pytest.raises(ValueError):
        validate_grayscale_image(image)


@pytest.mark.parametrize(
    "threshold,expected_error",
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (True, TypeError),
        ("0.05", TypeError),
    ],
)
def test_invalid_threshold(
    threshold: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        find_foreground_bounding_box(
            np.zeros((4, 4)),
            threshold=threshold,
        )


@pytest.mark.parametrize(
    "padding,expected_error",
    [
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_invalid_padding(
    padding: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        crop_to_foreground(
            np.zeros((4, 4)),
            padding=padding,
        )
