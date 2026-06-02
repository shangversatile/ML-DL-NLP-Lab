"""Adam optimizer."""

from numbers import Real

import numpy as np


class Adam:
    """
    Adam optimizer with first-moment, second-moment, and bias-correction state.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if not isinstance(learning_rate, Real):
            raise TypeError("learning_rate must be numeric.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not isinstance(beta1, Real):
            raise TypeError("beta1 must be numeric.")
        if beta1 < 0 or beta1 >= 1:
            raise ValueError("beta1 must satisfy 0 <= beta1 < 1.")
        if not isinstance(beta2, Real):
            raise TypeError("beta2 must be numeric.")
        if beta2 < 0 or beta2 >= 1:
            raise ValueError("beta2 must satisfy 0 <= beta2 < 1.")
        if not isinstance(epsilon, Real):
            raise TypeError("epsilon must be numeric.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment_weights = None
        self.second_moment_weights = None
        self.first_moment_bias = 0.0
        self.second_moment_bias = 0.0
        self.time_step = 0

    def step(
        self,
        weights: np.ndarray,
        bias: float,
        weight_gradients: np.ndarray,
        bias_gradient: float,
    ) -> tuple[np.ndarray, float]:
        """Return updated weights and bias using Adam."""
        if not isinstance(weights, np.ndarray):
            raise TypeError("weights must be a NumPy array.")
        if not isinstance(weight_gradients, np.ndarray):
            raise TypeError("weight_gradients must be a NumPy array.")
        if weights.shape != weight_gradients.shape:
            raise ValueError("weights and weight_gradients must have the same shape.")
        if not isinstance(bias, Real):
            raise TypeError("bias must be numeric.")
        if not isinstance(bias_gradient, Real):
            raise TypeError("bias_gradient must be numeric.")

        if self.first_moment_weights is None:
            self.first_moment_weights = np.zeros_like(weights)
        if self.second_moment_weights is None:
            self.second_moment_weights = np.zeros_like(weights)

        self.time_step += 1

        self.first_moment_weights = (
            self.beta1 * self.first_moment_weights
            + (1 - self.beta1) * weight_gradients
        )
        self.second_moment_weights = (
            self.beta2 * self.second_moment_weights
            + (1 - self.beta2) * (weight_gradients**2)
        )
        self.first_moment_bias = (
            self.beta1 * self.first_moment_bias
            + (1 - self.beta1) * bias_gradient
        )
        self.second_moment_bias = (
            self.beta2 * self.second_moment_bias
            + (1 - self.beta2) * (bias_gradient**2)
        )

        corrected_first_moment_weights = self.first_moment_weights / (
            1 - self.beta1**self.time_step
        )
        corrected_second_moment_weights = self.second_moment_weights / (
            1 - self.beta2**self.time_step
        )
        corrected_first_moment_bias = self.first_moment_bias / (
            1 - self.beta1**self.time_step
        )
        corrected_second_moment_bias = self.second_moment_bias / (
            1 - self.beta2**self.time_step
        )

        new_weights = weights - self.learning_rate * corrected_first_moment_weights / (
            np.sqrt(corrected_second_moment_weights) + self.epsilon
        )
        new_bias = bias - self.learning_rate * corrected_first_moment_bias / (
            np.sqrt(corrected_second_moment_bias) + self.epsilon
        )

        return new_weights, float(new_bias)
