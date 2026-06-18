import numpy as np
import pytest

from src.data.digits import load_digits_dataset
from src.data.preprocessing import stratified_train_val_test_split


def test_load_digits_dataset_scaled_data_shapes_and_ranges() -> None:
    X, y, images = load_digits_dataset(scale_pixels=True)

    assert X.ndim == 2
    assert X.shape[1] == 64
    assert images.ndim == 3
    assert images.shape[1:] == (8, 8)
    assert X.shape[0] == y.shape[0] == images.shape[0]
    assert np.all((y >= 0) & (y <= 9))
    assert X.min() >= 0.0
    assert X.max() <= 1.0
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_load_digits_dataset_unscaled_data_range() -> None:
    X, _, images = load_digits_dataset(scale_pixels=False)

    assert X.max() > 1.0
    assert X.max() <= 16.0
    assert images.max() > 1.0
    assert images.max() <= 16.0


def test_stratified_split_shapes_and_sample_counts() -> None:
    X, y, _ = load_digits_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = (
        stratified_train_val_test_split(
            X,
            y,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
    )

    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(y_train) + len(y_val) + len(y_test) == len(y)
    assert X_train.shape[1] == X.shape[1]
    assert X_val.shape[1] == X.shape[1]
    assert X_test.shape[1] == X.shape[1]
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)


def test_stratified_split_every_split_contains_all_classes() -> None:
    X, y, _ = load_digits_dataset()

    _, _, _, y_train, y_val, y_test = stratified_train_val_test_split(
        X,
        y,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert set(y_train) == set(range(10))
    assert set(y_val) == set(range(10))
    assert set(y_test) == set(range(10))


def test_stratified_split_reproducibility() -> None:
    X, y, _ = load_digits_dataset()

    first_split = stratified_train_val_test_split(
        X,
        y,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    second_split = stratified_train_val_test_split(
        X,
        y,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    for first_array, second_array in zip(first_split, second_split):
        np.testing.assert_array_equal(first_array, second_array)


def test_stratified_split_different_seeds_change_train_ordering() -> None:
    X, y, _ = load_digits_dataset()

    X_train_first, _, _, _, _, _ = stratified_train_val_test_split(
        X,
        y,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    X_train_second, _, _, _, _, _ = stratified_train_val_test_split(
        X,
        y,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=43,
    )

    assert not np.array_equal(X_train_first, X_train_second)


def test_stratified_split_does_not_mutate_original_arrays() -> None:
    X, y, _ = load_digits_dataset()
    original_X = X.copy()
    original_y = y.copy()

    stratified_train_val_test_split(
        X,
        y,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


def test_stratified_split_invalid_array_arguments() -> None:
    X = np.arange(40).reshape(20, 2)
    y = np.array([0, 1] * 10)

    with pytest.raises(TypeError):
        stratified_train_val_test_split(
            X.tolist(),
            y,
            val_ratio=0.2,
            test_ratio=0.2,
        )

    with pytest.raises(TypeError):
        stratified_train_val_test_split(
            X,
            y.tolist(),
            val_ratio=0.2,
            test_ratio=0.2,
        )

    with pytest.raises(ValueError):
        stratified_train_val_test_split(
            X.reshape(20, 1, 2),
            y,
            val_ratio=0.2,
            test_ratio=0.2,
        )

    with pytest.raises(ValueError):
        stratified_train_val_test_split(
            X,
            y.reshape(20, 1),
            val_ratio=0.2,
            test_ratio=0.2,
        )

    with pytest.raises(ValueError):
        stratified_train_val_test_split(
            X,
            y[:-1],
            val_ratio=0.2,
            test_ratio=0.2,
        )


@pytest.mark.parametrize(
    "val_ratio,test_ratio,expected_error",
    [
        (0.0, 0.2, ValueError),
        (-0.1, 0.2, ValueError),
        (1.2, 0.2, ValueError),
        (0.5, 0.5, ValueError),
        (True, 0.2, TypeError),
        (0.2, False, TypeError),
        ("0.2", 0.2, TypeError),
        (0.2, "0.2", TypeError),
    ],
)
def test_stratified_split_invalid_ratios(
    val_ratio: object,
    test_ratio: object,
    expected_error: type[Exception],
) -> None:
    X = np.arange(40).reshape(20, 2)
    y = np.array([0, 1] * 10)

    with pytest.raises(expected_error):
        stratified_train_val_test_split(
            X,
            y,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
