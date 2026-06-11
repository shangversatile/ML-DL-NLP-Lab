"""Dataset generation utilities."""

from numbers import Real

import numpy as np


def _validate_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_real(name: str, value: float) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric and not boolean")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


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
    Generate a roughly balanced linearly separable binary classification dataset.

    Labels are created by thresholding logits at the median value.

    Return:
    - X: shape (n_samples, n_features)
    - y: shape (n_samples,), values should be 0 or 1
    """
    _validate_positive_int("n_samples", n_samples)
    _validate_positive_int("n_features", n_features)

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    true_weights = rng.standard_normal(n_features)

    logits = X @ true_weights
    threshold = np.median(logits)
    y = (logits >= threshold).astype(int)

    return X, y


def make_xor_classification_data(
    n_samples: int = 400,
    noise: float = 0.15,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a noisy two-dimensional XOR-style binary classification dataset.

    Class rule before feature noise:
    y = 1 when x1 and x2 have the same sign.
    y = 0 otherwise.
    """
    _validate_positive_int("n_samples", n_samples)
    _validate_non_negative_real("noise", noise)

    rng = np.random.default_rng(seed)
    X_clean = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    y = ((X_clean[:, 0] * X_clean[:, 1]) > 0).astype(int)
    X = X_clean + rng.normal(
        loc=0.0,
        scale=noise,
        size=X_clean.shape,
    )

    return X, y
