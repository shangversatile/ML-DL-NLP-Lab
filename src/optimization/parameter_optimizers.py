"""Generic optimizers for dictionaries of NumPy parameter arrays."""

from numbers import Real

import numpy as np


ParameterDict = dict[str, np.ndarray]


def _validate_positive_real(name: str, value: float) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric and not boolean.")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_beta(name: str, value: float) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric and not boolean.")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    if value < 0 or value >= 1:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")


def _validate_finite_array(name: str, array: np.ndarray) -> None:
    try:
        is_finite = np.all(np.isfinite(array))
    except TypeError as error:
        raise ValueError(f"{name} must contain only finite values.") from error

    if not is_finite:
        raise ValueError(f"{name} must contain only finite values.")


def _validate_parameter_gradient_dicts(
    parameters: ParameterDict,
    gradients: ParameterDict,
) -> None:
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dictionary.")
    if not isinstance(gradients, dict):
        raise TypeError("gradients must be a dictionary.")
    if not parameters:
        raise ValueError("parameters must not be empty.")

    for key in parameters:
        if not isinstance(key, str):
            raise TypeError("parameter keys must be strings.")
    for key in gradients:
        if not isinstance(key, str):
            raise TypeError("gradient keys must be strings.")

    if set(parameters) != set(gradients):
        raise ValueError("parameters and gradients must have matching keys.")

    for name in parameters:
        parameter = parameters[name]
        gradient = gradients[name]

        if not isinstance(parameter, np.ndarray):
            raise TypeError(f"parameters[{name!r}] must be a NumPy array.")
        if not isinstance(gradient, np.ndarray):
            raise TypeError(f"gradients[{name!r}] must be a NumPy array.")
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"gradients[{name!r}] must match the parameter shape."
            )

        _validate_finite_array(f"parameters[{name!r}]", parameter)
        _validate_finite_array(f"gradients[{name!r}]", gradient)


def _validate_state_compatibility(
    parameters: ParameterDict,
    state: ParameterDict,
    state_name: str,
) -> None:
    if set(state) != set(parameters):
        raise ValueError(f"{state_name} keys must match parameter keys.")

    for name, parameter in parameters.items():
        if not isinstance(state[name], np.ndarray):
            raise ValueError(f"{state_name}[{name!r}] must be a NumPy array.")
        if state[name].shape != parameter.shape:
            raise ValueError(
                f"{state_name}[{name!r}] must match the parameter shape."
            )


class ParameterSGD:
    """
    SGD-style updates for a dictionary of NumPy parameter arrays.

    Full-batch versus mini-batch behavior is controlled by the training loop,
    not by this optimizer class.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        _validate_positive_real("learning_rate", learning_rate)
        self.learning_rate = learning_rate

    def step(
        self,
        parameters: ParameterDict,
        gradients: ParameterDict,
    ) -> ParameterDict:
        _validate_parameter_gradient_dicts(parameters, gradients)

        return {
            name: parameters[name] - self.learning_rate * gradients[name]
            for name in parameters
        }


class ParameterMomentum:
    """
    Momentum updates for a dictionary of NumPy parameter arrays.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta: float = 0.9,
    ) -> None:
        _validate_positive_real("learning_rate", learning_rate)
        _validate_beta("beta", beta)

        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity: ParameterDict = {}

    def step(
        self,
        parameters: ParameterDict,
        gradients: ParameterDict,
    ) -> ParameterDict:
        _validate_parameter_gradient_dicts(parameters, gradients)

        if not self.velocity:
            self.velocity = {
                name: np.zeros_like(parameter, dtype=float)
                for name, parameter in parameters.items()
            }
        else:
            _validate_state_compatibility(parameters, self.velocity, "velocity")

        updated_parameters = {}
        for name in parameters:
            self.velocity[name] = (
                self.beta * self.velocity[name]
                + (1 - self.beta) * gradients[name]
            )
            updated_parameters[name] = (
                parameters[name] - self.learning_rate * self.velocity[name]
            )

        return updated_parameters


class ParameterAdam:
    """
    Adam updates for a dictionary of NumPy parameter arrays.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        _validate_positive_real("learning_rate", learning_rate)
        _validate_beta("beta1", beta1)
        _validate_beta("beta2", beta2)
        _validate_positive_real("epsilon", epsilon)

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment: ParameterDict = {}
        self.second_moment: ParameterDict = {}
        self.time_step = 0

    def step(
        self,
        parameters: ParameterDict,
        gradients: ParameterDict,
    ) -> ParameterDict:
        _validate_parameter_gradient_dicts(parameters, gradients)

        if not self.first_moment and not self.second_moment:
            self.first_moment = {
                name: np.zeros_like(parameter, dtype=float)
                for name, parameter in parameters.items()
            }
            self.second_moment = {
                name: np.zeros_like(parameter, dtype=float)
                for name, parameter in parameters.items()
            }
        else:
            _validate_state_compatibility(
                parameters,
                self.first_moment,
                "first_moment",
            )
            _validate_state_compatibility(
                parameters,
                self.second_moment,
                "second_moment",
            )

        self.time_step += 1

        updated_parameters = {}
        for name in parameters:
            self.first_moment[name] = (
                self.beta1 * self.first_moment[name]
                + (1 - self.beta1) * gradients[name]
            )
            self.second_moment[name] = (
                self.beta2 * self.second_moment[name]
                + (1 - self.beta2) * (gradients[name] ** 2)
            )

            corrected_first_moment = self.first_moment[name] / (
                1 - self.beta1 ** self.time_step
            )
            corrected_second_moment = self.second_moment[name] / (
                1 - self.beta2 ** self.time_step
            )

            updated_parameters[name] = (
                parameters[name]
                - self.learning_rate
                * corrected_first_moment
                / (np.sqrt(corrected_second_moment) + self.epsilon)
            )

        return updated_parameters
