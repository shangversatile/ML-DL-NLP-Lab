"""Reusable inference helpers for the scratch handwritten-digit MLP."""

from numbers import Integral

import numpy as np


def prepare_digit_features(
    inputs: np.ndarray,
) -> np.ndarray:
    """
    Convert digit inputs into a 2D feature matrix of shape (n_samples, 64).
    """
    if not isinstance(inputs, np.ndarray):
        raise TypeError("inputs must be a NumPy array.")
    try:
        is_finite = np.all(np.isfinite(inputs))
    except TypeError as error:
        raise ValueError("inputs must contain only finite numeric values.") from error
    if not is_finite:
        raise ValueError("inputs must contain only finite values.")

    features = np.array(inputs, dtype=float, copy=True)

    if features.shape == (64,):
        return features.reshape(1, 64)
    if features.shape == (8, 8):
        return features.reshape(1, 64)
    if features.ndim == 2 and features.shape[1] == 64:
        return features
    if features.ndim == 3 and features.shape[1:] == (8, 8):
        return features.reshape(features.shape[0], 64)

    raise ValueError(
        "inputs must have shape (64,), (8, 8), (n_samples, 64), "
        "or (n_samples, 8, 8)."
    )


def _validate_top_k(
    top_k: int,
    num_classes: int,
) -> None:
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, Integral):
        raise TypeError("top_k must be an integer.")
    if top_k < 1 or top_k > num_classes:
        raise ValueError("top_k must satisfy 1 <= top_k <= num_classes.")


def _validate_probability_matrix(
    probabilities: np.ndarray,
) -> None:
    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities must be a NumPy array.")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional.")
    if probabilities.shape[0] == 0:
        raise ValueError("probabilities must contain at least one sample.")
    if probabilities.shape[1] < 2:
        raise ValueError("probabilities must contain at least two classes.")
    if np.issubdtype(probabilities.dtype, np.bool_) or not np.issubdtype(
        probabilities.dtype,
        np.number,
    ):
        raise ValueError("probabilities must contain numeric values.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative.")

    row_sums = np.sum(probabilities, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("probability rows must sum to one.")


def predict_digit_batch(
    model,
    inputs: np.ndarray,
    top_k: int = 3,
) -> dict[str, np.ndarray]:
    """
    Run batch inference and return predictions, confidences, and top-k candidates.
    """
    X = prepare_digit_features(inputs)
    probabilities = model.predict_proba(X)
    _validate_probability_matrix(probabilities)
    _validate_top_k(top_k, probabilities.shape[1])

    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    top_k_indices = np.argsort(-probabilities, axis=1)[:, :top_k]
    top_k_probabilities = np.take_along_axis(
        probabilities,
        top_k_indices,
        axis=1,
    )

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "confidences": confidences,
        "top_k_indices": top_k_indices,
        "top_k_probabilities": top_k_probabilities,
    }


def predict_single_digit(
    model,
    digit_input: np.ndarray,
    top_k: int = 3,
) -> dict[str, np.ndarray | int | float]:
    """
    Run inference for one digit input.
    """
    results = predict_digit_batch(model, digit_input, top_k=top_k)

    if results["probabilities"].shape[0] != 1:
        raise ValueError("digit_input must contain exactly one digit.")

    return {
        "probabilities": results["probabilities"][0],
        "prediction": int(results["predictions"][0]),
        "confidence": float(results["confidences"][0]),
        "top_k_indices": results["top_k_indices"][0],
        "top_k_probabilities": results["top_k_probabilities"][0],
    }
