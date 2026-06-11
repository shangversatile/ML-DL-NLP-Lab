import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import (
    make_binary_classification_data,
    make_linear_regression_data,
    make_xor_classification_data,
)
from src.data.preprocessing import iterate_minibatches, standardize_features, train_val_split


def test_linear_regression_data_shapes():
    X, y, true_weights, true_bias = make_linear_regression_data(
        n_samples=50,
        n_features=3,
        seed=42,
    )

    assert X.shape == (50, 3)
    assert y.shape == (50,)
    assert true_weights.shape == (3,)
    assert isinstance(true_bias, float)


def test_binary_classification_labels():
    X, y = make_binary_classification_data(
        n_samples=50,
        n_features=4,
        seed=42,
    )

    assert X.shape == (50, 4)
    assert set(np.unique(y)).issubset({0, 1})


def test_binary_classification_contains_both_classes():
    _, y = make_binary_classification_data(
        n_samples=100,
        n_features=2,
        seed=42,
    )

    assert set(np.unique(y)) == {0, 1}


def test_binary_classification_is_roughly_balanced():
    _, y = make_binary_classification_data(
        n_samples=200,
        n_features=2,
        seed=42,
    )

    positive_rate = y.mean()
    assert 0.4 <= positive_rate <= 0.6


def test_xor_classification_data_shapes_and_labels():
    X, y = make_xor_classification_data(
        n_samples=100,
        noise=0.1,
        seed=42,
    )

    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert set(np.unique(y)).issubset({0, 1})
    assert set(np.unique(y)) == {0, 1}


def test_xor_classification_data_reproducibility():
    X_first, y_first = make_xor_classification_data(
        n_samples=100,
        noise=0.1,
        seed=42,
    )
    X_second, y_second = make_xor_classification_data(
        n_samples=100,
        noise=0.1,
        seed=42,
    )

    assert np.array_equal(X_first, X_second)
    assert np.array_equal(y_first, y_second)


def test_xor_classification_data_different_seeds():
    X_first, _ = make_xor_classification_data(
        n_samples=100,
        noise=0.1,
        seed=42,
    )
    X_second, _ = make_xor_classification_data(
        n_samples=100,
        noise=0.1,
        seed=43,
    )

    assert not np.array_equal(X_first, X_second)


def test_xor_classification_data_invalid_arguments():
    with pytest.raises(ValueError):
        make_xor_classification_data(n_samples=0)

    with pytest.raises(ValueError):
        make_xor_classification_data(noise=-0.1)

    with pytest.raises(TypeError):
        make_xor_classification_data(noise=True)


def test_linear_regression_data_reproducibility():
    X_first, y_first, _, _ = make_linear_regression_data(seed=42)
    X_second, y_second, _, _ = make_linear_regression_data(seed=42)

    assert np.array_equal(X_first, X_second)
    assert np.array_equal(y_first, y_second)


def test_train_val_split_shapes():
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)

    X_train, X_val, y_train, y_val = train_val_split(X, y, val_ratio=0.2, seed=42)

    assert X_train.shape == (80, 2)
    assert X_val.shape == (20, 2)
    assert y_train.shape == (80,)
    assert y_val.shape == (20,)


def test_iterate_minibatches_without_shuffle():
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3, 4])

    batches = list(iterate_minibatches(X, y, batch_size=2, shuffle=False))

    assert len(batches) == 3
    assert batches[0][0].shape == (2, 2)
    assert batches[1][0].shape == (2, 2)
    assert batches[2][0].shape == (1, 2)
    assert np.array_equal(batches[0][0], np.array([[0, 1], [2, 3]]))
    assert np.array_equal(batches[0][1], np.array([0, 1]))
    assert np.array_equal(batches[1][0], np.array([[4, 5], [6, 7]]))
    assert np.array_equal(batches[1][1], np.array([2, 3]))
    assert np.array_equal(batches[2][0], np.array([[8, 9]]))
    assert np.array_equal(batches[2][1], np.array([4]))


def test_iterate_minibatches_covers_all_samples_once():
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3, 4])

    batches = list(iterate_minibatches(X, y, batch_size=2, shuffle=True, seed=42))
    yielded_y = np.concatenate([batch_y for _, batch_y in batches])

    assert np.array_equal(np.sort(yielded_y), y)
    assert len(yielded_y) == len(np.unique(yielded_y))


def test_iterate_minibatches_reproducible_shuffling():
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3, 4])

    first_batches = list(iterate_minibatches(X, y, batch_size=2, shuffle=True, seed=42))
    second_batches = list(iterate_minibatches(X, y, batch_size=2, shuffle=True, seed=42))
    first_order = np.concatenate([batch_y for _, batch_y in first_batches])
    second_order = np.concatenate([batch_y for _, batch_y in second_batches])

    assert np.array_equal(first_order, second_order)


def test_iterate_minibatches_different_shuffle_seeds():
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3, 4])

    first_batches = list(iterate_minibatches(X, y, batch_size=2, shuffle=True, seed=42))
    second_batches = list(iterate_minibatches(X, y, batch_size=2, shuffle=True, seed=43))
    first_order = np.concatenate([batch_y for _, batch_y in first_batches])
    second_order = np.concatenate([batch_y for _, batch_y in second_batches])

    assert not np.array_equal(first_order, second_order)


def test_iterate_minibatches_batch_size_larger_than_dataset():
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3, 4])

    batches = list(iterate_minibatches(X, y, batch_size=10, shuffle=False))

    assert len(batches) == 1
    assert np.array_equal(batches[0][0], X)
    assert np.array_equal(batches[0][1], y)


def test_iterate_minibatches_invalid_batch_size():
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3, 4])

    with pytest.raises(ValueError):
        list(iterate_minibatches(X, y, batch_size=0))

    with pytest.raises(ValueError):
        list(iterate_minibatches(X, y, batch_size=-1))

    with pytest.raises(ValueError):
        list(iterate_minibatches(X, y, batch_size=1.5))


def test_iterate_minibatches_mismatched_sample_counts():
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3])

    with pytest.raises(ValueError):
        list(iterate_minibatches(X, y, batch_size=2))


def test_standardize_features():
    X_train = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )

    X_train_scaled, X_val_scaled, mean, std = standardize_features(X_train)

    assert X_val_scaled is None
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0)
    assert np.allclose(X_train_scaled.std(axis=0), 1.0)
