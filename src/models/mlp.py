"""Multi-layer perceptron implementation from scratch."""

from __future__ import annotations

import numpy as np


class BinaryMLPScratch:
    """
    One-hidden-layer MLP for binary classification implemented with NumPy.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        seed: int | None = None,
    ) -> None:
        if type(n_features) is not int:
            raise TypeError("n_features must be an integer.")
        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        if type(hidden_dim) is not int:
            raise TypeError("hidden_dim must be an integer.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.n_features = n_features
        self.hidden_dim = hidden_dim

        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_features, hidden_dim)) * np.sqrt(
            2.0 / n_features
        )
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, 1)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities for each value in z."""
        z = np.asarray(z)
        probabilities = np.empty_like(z, dtype=float)

        positive_mask = z >= 0
        probabilities[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))

        exp_z = np.exp(z[~positive_mask])
        probabilities[~positive_mask] = exp_z / (1.0 + exp_z)

        return probabilities

    def forward(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute probabilities and return a cache for future backpropagation.
        """
        self._validate_X(X)

        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        probabilities = self._sigmoid(Z2).reshape(-1)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
        }
        return probabilities, cache

    def _validate_X(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D NumPy array.")
        if X.shape[1] != self.n_features:
            raise ValueError("X must have the same number of columns as n_features.")
