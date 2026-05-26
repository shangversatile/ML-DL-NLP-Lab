"""Linear regression implementation from scratch."""

import numpy as np


class LinearRegressionScratch:
    """Linear regression model implemented from scratch with NumPy."""

    def __init__(self, n_features: int) -> None:
        if not isinstance(n_features, int):
            raise TypeError("n_features must be an integer.")
        if n_features <= 0:
            raise ValueError("n_features must be positive.")

        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError("X must have the same number of columns as weights.")

        return X @ self.weights + self.bias

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        self._validate_targets(y, X.shape[0])

        loss = np.mean((predictions - y) ** 2)
        return float(loss)

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        predictions = self.predict(X)
        self._validate_targets(y, X.shape[0])

        error = predictions - y
        n_samples = X.shape[0]
        dw = (2 / n_samples) * X.T @ error
        db = (2 / n_samples) * np.sum(error)

        return dw, float(db)

    @staticmethod
    def _validate_targets(y: np.ndarray, n_samples: int) -> None:
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        if y.shape != (n_samples,):
            raise ValueError("y must have shape (n_samples,).")
