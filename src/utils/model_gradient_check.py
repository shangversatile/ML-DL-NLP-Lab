"""Generic numerical gradient checking utilities for scratch NumPy models."""

from numbers import Real

import numpy as np


PARAMETER_TO_GRADIENT_KEY = {
    "W1": "dW1",
    "b1": "db1",
    "W2": "dW2",
    "b2": "db2",
}


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
    if not np.all(np.isfinite(analytical_gradient)):
        raise ValueError("analytical_gradient values must be finite.")
    if not np.all(np.isfinite(numerical_gradient)):
        raise ValueError("numerical_gradient values must be finite.")
    _validate_positive_real("stability_constant", stability_constant)

    numerator = np.linalg.norm(analytical_gradient - numerical_gradient)
    denominator = (
        np.linalg.norm(analytical_gradient)
        + np.linalg.norm(numerical_gradient)
        + stability_constant
    )
    return float(numerator / denominator)


def compute_numerical_gradients(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-5,
    parameter_to_gradient_key: dict[str, str] = PARAMETER_TO_GRADIENT_KEY,
) -> dict[str, np.ndarray]:
    """
    Approximate model parameter gradients with central finite differences.
    """
    _validate_positive_real("epsilon", epsilon)
    if not isinstance(parameter_to_gradient_key, dict):
        raise TypeError("parameter_to_gradient_key must be a dictionary.")

    original_parameters = model.get_parameters()
    if set(parameter_to_gradient_key) != set(original_parameters):
        raise ValueError("parameter_to_gradient_key must match model parameters.")

    numerical_gradients = {}
    try:
        for name, gradient_key in parameter_to_gradient_key.items():
            parameter = original_parameters[name]
            numerical_gradient = np.zeros_like(parameter, dtype=float)

            for index in np.ndindex(parameter.shape):
                plus_parameters = {
                    parameter_name: parameter_value.copy()
                    for parameter_name, parameter_value in original_parameters.items()
                }
                minus_parameters = {
                    parameter_name: parameter_value.copy()
                    for parameter_name, parameter_value in original_parameters.items()
                }

                plus_parameters[name][index] += epsilon
                minus_parameters[name][index] -= epsilon

                model.set_parameters(plus_parameters)
                loss_plus = model.compute_loss(X, y)

                model.set_parameters(minus_parameters)
                loss_minus = model.compute_loss(X, y)

                numerical_gradient[index] = (loss_plus - loss_minus) / (
                    2.0 * epsilon
                )

            numerical_gradients[gradient_key] = numerical_gradient
    finally:
        model.set_parameters(original_parameters)

    return numerical_gradients


def compare_gradients(
    analytical_gradients: dict[str, np.ndarray],
    numerical_gradients: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compute relative L2 error for matching analytical and numerical gradients.
    """
    if not isinstance(analytical_gradients, dict):
        raise TypeError("analytical_gradients must be a dictionary.")
    if not isinstance(numerical_gradients, dict):
        raise TypeError("numerical_gradients must be a dictionary.")
    if set(analytical_gradients) != set(numerical_gradients):
        raise ValueError("gradient dictionaries must have matching keys.")

    errors = {}
    for gradient_key in analytical_gradients:
        if analytical_gradients[gradient_key].shape != numerical_gradients[
            gradient_key
        ].shape:
            raise ValueError("gradient shapes must match.")
        errors[gradient_key] = relative_l2_error(
            analytical_gradients[gradient_key],
            numerical_gradients[gradient_key],
        )

    return errors


def _validate_positive_real(name: str, value: float) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive real number.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
