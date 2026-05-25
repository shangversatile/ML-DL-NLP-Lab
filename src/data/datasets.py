"""Dataset generation utilities."""

import numpy as np


def _validate_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def make_linear_regression_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate a synthetic linear regression dataset.

    Return:
    - X: shape (n_samples, n_features)
    - y: shape (n_samples,)
    - true_weights: shape (n_features,)
    - true_bias: float

    Data generation rule:
    y = X @ true_weights + true_bias + Gaussian noise
    """
    _validate_positive_int("n_samples", n_samples)
    _validate_positive_int("n_features", n_features)
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    true_weights = rng.standard_normal(n_features)
    true_bias = float(rng.standard_normal())
    noise_values = rng.normal(loc=0.0, scale=noise, size=n_samples)
    y = X @ true_weights + true_bias + noise_values

    return X, y, true_weights, true_bias


def make_binary_classification_data(
    n_samples: int = 100,
    n_features: int = 2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple synthetic binary classification dataset.

    Return:
    - X: shape (n_samples, n_features)
    - y: shape (n_samples,), values should be 0 or 1
    """
    _validate_positive_int("n_samples", n_samples)
    _validate_positive_int("n_features", n_features)

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    true_weights = rng.standard_normal(n_features)
    true_bias = float(rng.standard_normal())

    logits = X @ true_weights + true_bias
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = (probabilities > 0.5).astype(int)

    return X, y
