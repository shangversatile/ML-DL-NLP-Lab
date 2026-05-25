import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import (
    make_binary_classification_data,
    make_linear_regression_data,
)
from src.data.preprocessing import standardize_features, train_val_split


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
