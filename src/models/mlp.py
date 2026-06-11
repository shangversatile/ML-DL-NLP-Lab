"""Multi-layer perceptron implementation from scratch."""

from __future__ import annotations

import numpy as np


class BinaryMLPScratch:
    """
    One-hidden-layer MLP for binary classification implemented with NumPy.
    """

    PARAMETER_NAMES = ("W1", "b1", "W2", "b2")

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

    def get_parameters(self) -> dict[str, np.ndarray]:
        """
        Return copies of all trainable parameters.
        """
        return {
            name: getattr(self, name).copy()
            for name in self.PARAMETER_NAMES
        }

    def set_parameters(
        self,
        parameters: dict[str, np.ndarray],
    ) -> None:
        """
        Replace trainable parameters after validating keys and shapes.
        """
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary.")
        if set(parameters) != set(self.PARAMETER_NAMES):
            raise ValueError("parameters must contain W1, b1, W2, and b2.")

        for name in self.PARAMETER_NAMES:
            replacement = parameters[name]
            current_parameter = getattr(self, name)

            if not isinstance(replacement, np.ndarray):
                raise TypeError(f"parameters[{name!r}] must be a NumPy array.")
            if replacement.shape != current_parameter.shape:
                raise ValueError(
                    f"parameters[{name!r}] must match the current parameter shape."
                )
            if not np.all(np.isfinite(replacement)):
                raise ValueError(
                    f"parameters[{name!r}] must contain only finite values."
                )

        for name in self.PARAMETER_NAMES:
            setattr(self, name, parameters[name].copy())

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

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return positive-class probabilities.
        """
        probabilities, _ = self.forward(X)
        return probabilities

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Return thresholded binary predictions.
        """
        if isinstance(threshold, (bool, np.bool_)):
            raise TypeError("threshold must be numeric and not boolean.")
        if not isinstance(threshold, (int, float, np.integer, np.floating)):
            raise TypeError("threshold must be numeric.")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")

        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Compute mean binary cross entropy loss.
        """
        probabilities, _ = self.forward(X)
        self._validate_y(y, X.shape[0])

        epsilon = 1e-15
        clipped_probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
        loss = -np.mean(
            y * np.log(clipped_probabilities)
            + (1 - y) * np.log(1.0 - clipped_probabilities)
        )

        return float(loss)

    def compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Compute analytical gradients using backpropagation.
        """
        probabilities, cache = self.forward(X)
        self._validate_y(y, X.shape[0])

        n_samples = X.shape[0]
        y_column = y.reshape(-1, 1)
        probabilities_column = probabilities.reshape(-1, 1)

        dZ2 = (probabilities_column - y_column) / n_samples
        dW2 = cache["A1"].T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (cache["Z1"] > 0)

        dW1 = cache["X"].T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }

    def _validate_X(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D NumPy array.")
        if X.shape[1] != self.n_features:
            raise ValueError("X must have the same number of columns as n_features.")

    def _validate_y(
        self,
        y: np.ndarray,
        n_samples: int,
    ) -> None:
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        if y.shape != (n_samples,):
            raise ValueError("y must have shape (n_samples,).")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain only 0 and 1.")
