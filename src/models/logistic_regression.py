"""Logistic regression implementation from scratch."""

from __future__ import annotations

from numbers import Real

import numpy as np


class LogisticRegressionScratch:
    """
    Binary logistic regression model implemented from scratch with NumPy.
    """

    def __init__(self, n_features: int) -> None:
        if type(n_features) is not int:
            raise TypeError("n_features must be an integer.")
        if n_features <= 0:
            raise ValueError("n_features must be positive.")

        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities for each value in z."""
        z = np.asarray(z)
        probabilities = np.empty_like(z, dtype=float)

        positive_mask = z >= 0
        probabilities[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))

        exp_z = np.exp(z[~positive_mask])
        probabilities[~positive_mask] = exp_z / (1.0 + exp_z)

        return probabilities

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the positive class."""
        self._validate_X(X)

        logits = X @ self.weights + self.bias
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        if not isinstance(threshold, Real):
            raise TypeError("threshold must be numeric.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross entropy loss."""
        self._validate_X(X)
        self._validate_y(y, X.shape[0])

        probabilities = self.predict_proba(X)
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)

        loss = -np.mean(
            y * np.log(probabilities) + (1 - y) * np.log(1.0 - probabilities)
        )
        return float(loss)

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute analytical gradients for weights and bias."""
        self._validate_X(X)
        self._validate_y(y, X.shape[0])

        probabilities = self.predict_proba(X)
        error = probabilities - y
        n_samples = X.shape[0]

        dw = (1.0 / n_samples) * (X.T @ error)
        db = (1.0 / n_samples) * np.sum(error)

        return dw, float(db)

    def _validate_X(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D NumPy array.")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError("X must have the same number of columns as weights.")

    def _validate_y(self, y: np.ndarray, n_samples: int) -> None:
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        if y.shape != (n_samples,):
            raise ValueError("y must have shape (n_samples,).")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain only 0 and 1 values.")
