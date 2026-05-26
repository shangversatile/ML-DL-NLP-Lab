"""Gradient descent optimizer."""

from numbers import Real

import numpy as np


class BatchGradientDescent:
    """Basic batch gradient descent optimizer."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        if not isinstance(learning_rate, Real):
            raise TypeError("learning_rate must be numeric.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.learning_rate = learning_rate

    def step(
        self,
        weights: np.ndarray,
        bias: float,
        weight_gradients: np.ndarray,
        bias_gradient: float,
    ) -> tuple[np.ndarray, float]:
        """Return updated weights and bias using batch gradient descent."""
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

        new_weights = weights - self.learning_rate * weight_gradients
        new_bias = bias - self.learning_rate * bias_gradient

        return new_weights, float(new_bias)
