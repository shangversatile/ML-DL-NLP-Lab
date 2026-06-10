"""Numerical gradient checking utilities."""

from numbers import Real

import numpy as np


MLP_PARAMETER_NAMES = ("W1", "b1", "W2", "b2")


def _validate_positive_real(value: Real, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric and not boolean.")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def compute_numerical_gradients(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-5,
    parameter_names: tuple[str, ...] = MLP_PARAMETER_NAMES,
) -> dict[str, np.ndarray]:
    """
    Approximate parameter gradients with central finite differences.
    """
    _validate_positive_real(epsilon, "epsilon")

    numerical_gradients = {}
    for parameter_name in parameter_names:
        parameter = getattr(model, parameter_name)
        if not isinstance(parameter, np.ndarray):
            raise TypeError(f"{parameter_name} must be a NumPy array.")

        gradient = np.zeros_like(parameter, dtype=float)
        iterator = np.nditer(parameter, flags=["multi_index"], op_flags=["readwrite"])

        for parameter_entry in iterator:
            index = iterator.multi_index
            original_value = float(parameter_entry)

            try:
                parameter_entry[...] = original_value + epsilon
                loss_plus = model.compute_loss(X, y)

                parameter_entry[...] = original_value - epsilon
                loss_minus = model.compute_loss(X, y)
            finally:
                parameter_entry[...] = original_value

            gradient[index] = (loss_plus - loss_minus) / (2 * epsilon)

        numerical_gradients[parameter_name] = gradient

    return numerical_gradients


def relative_l2_error(
    analytical_gradient: np.ndarray,
    numerical_gradient: np.ndarray,
    stability_constant: float = 1e-12,
) -> float:
    """
    Compute relative L2 error between analytical and numerical gradients.
    """
    if not isinstance(analytical_gradient, np.ndarray):
        raise TypeError("analytical_gradient must be a NumPy array.")
    if not isinstance(numerical_gradient, np.ndarray):
        raise TypeError("numerical_gradient must be a NumPy array.")
    if analytical_gradient.shape != numerical_gradient.shape:
        raise ValueError("gradient shapes must match.")
    _validate_positive_real(stability_constant, "stability_constant")

    numerator = np.linalg.norm(analytical_gradient - numerical_gradient)
    denominator = (
        np.linalg.norm(analytical_gradient)
        + np.linalg.norm(numerical_gradient)
        + stability_constant
    )

    return float(numerator / denominator)


def _normalize_gradient_keys(
    gradients: dict[str, np.ndarray],
    target_keys: set[str],
) -> dict[str, np.ndarray]:
    if set(gradients) == target_keys:
        return gradients

    normalized = {}
    for key, value in gradients.items():
        parameter_key = key[1:] if key.startswith("d") else key
        normalized[parameter_key] = value

    if set(normalized) != target_keys:
        raise ValueError("gradient dictionaries must have matching parameter keys.")

    return normalized


def compare_gradients(
    analytical_gradients: dict[str, np.ndarray],
    numerical_gradients: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compute relative L2 error for each parameter gradient.
    """
    if not isinstance(analytical_gradients, dict):
        raise TypeError("analytical_gradients must be a dictionary.")
    if not isinstance(numerical_gradients, dict):
        raise TypeError("numerical_gradients must be a dictionary.")

    numerical_keys = set(numerical_gradients)
    analytical_gradients = _normalize_gradient_keys(
        analytical_gradients,
        numerical_keys,
    )

    return {
        parameter_name: relative_l2_error(
            analytical_gradients[parameter_name],
            numerical_gradients[parameter_name],
        )
        for parameter_name in numerical_gradients
    }
