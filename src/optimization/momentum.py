"""Momentum optimizer."""

from numbers import Real

import numpy as np


class Momentum:
    """
    Momentum optimizer using an exponential moving average of gradients.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta: float = 0.9,
    ) -> None:
        if not isinstance(learning_rate, Real):
            raise TypeError("learning_rate must be numeric.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not isinstance(beta, Real):
            raise TypeError("beta must be numeric.")
        if beta < 0 or beta >= 1:
            raise ValueError("beta must satisfy 0 <= beta < 1.")

        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity_weights = None
        self.velocity_bias = 0.0

    def step(
        self,
        weights: np.ndarray,
        bias: float,
        weight_gradients: np.ndarray,
        bias_gradient: float,
    ) -> tuple[np.ndarray, float]:
        """Return updated weights and bias using Momentum."""
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

        if self.velocity_weights is None:
            self.velocity_weights = np.zeros_like(weights)

        self.velocity_weights = (
            self.beta * self.velocity_weights
            + (1 - self.beta) * weight_gradients
        )
        self.velocity_bias = (
            self.beta * self.velocity_bias
            + (1 - self.beta) * bias_gradient
        )

        new_weights = weights - self.learning_rate * self.velocity_weights
        new_bias = bias - self.learning_rate * self.velocity_bias

        return new_weights, float(new_bias)
