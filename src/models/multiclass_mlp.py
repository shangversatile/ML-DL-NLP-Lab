"""Multiclass MLP implemented from scratch with NumPy."""

from numbers import Integral, Real

import numpy as np

from src.utils.multiclass import (
    multiclass_cross_entropy,
    softmax_cross_entropy_gradient,
    stable_softmax,
)


class MulticlassMLPScratch:
    """
    Single-hidden-layer ReLU MLP for multiclass classification.
    """

    PARAMETER_NAMES = ("W1", "b1", "W2", "b2")

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        num_classes: int,
        seed: int | None = None,
    ) -> None:
        self._validate_positive_integer("n_features", n_features)
        self._validate_positive_integer("hidden_dim", hidden_dim)
        self._validate_num_classes(num_classes)

        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        rng = np.random.default_rng(seed)
        # W1 maps input features to hidden ReLU units. He-style scaling is used
        # because ReLU layers preserve variance better with this initialization.
        self.W1 = rng.standard_normal((self.n_features, self.hidden_dim)) * np.sqrt(
            2.0 / self.n_features
        )
        self.b1 = np.zeros(self.hidden_dim)
        # W2 maps hidden activations to class logits before softmax.
        self.W2 = rng.standard_normal((self.hidden_dim, self.num_classes)) * np.sqrt(
            2.0 / self.hidden_dim
        )
        self.b2 = np.zeros(self.num_classes)

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    @staticmethod
    def _validate_num_classes(num_classes: int) -> None:
        if isinstance(num_classes, (bool, np.bool_)) or not isinstance(
            num_classes,
            Integral,
        ):
            raise TypeError("num_classes must be an integer.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

    def _validate_X(
        self,
        X: np.ndarray,
    ) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D NumPy array.")
        if X.shape[1] != self.n_features:
            raise ValueError("X must have the same number of columns as n_features.")
        if np.issubdtype(X.dtype, np.bool_) or not np.issubdtype(X.dtype, np.number):
            raise ValueError("X values must be numeric.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X values must be finite.")

    def _validate_y(
        self,
        y: np.ndarray,
        n_samples: int,
    ) -> None:
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D NumPy array.")
        if y.shape[0] != n_samples:
            raise ValueError("y must have one label per sample.")
        if np.issubdtype(y.dtype, np.bool_) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer labels.")
        if np.any(y < 0) or np.any(y >= self.num_classes):
            raise ValueError("y labels must satisfy 0 <= label < num_classes.")

    @staticmethod
    def _relu(
        z: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(0.0, z)

    def forward(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Return class probabilities and forward-pass cache.
        """
        self._validate_X(X)

        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        logits = A1 @ self.W2 + self.b2
        probabilities = stable_softmax(logits)

        return probabilities, {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "logits": logits,
        }

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return class probabilities.
        """
        probabilities, _ = self.forward(X)
        return probabilities

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return predicted integer class labels.
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Compute mean multiclass cross entropy.
        """
        probabilities, _ = self.forward(X)
        self._validate_y(y, X.shape[0])
        return float(multiclass_cross_entropy(y, probabilities))

    def compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Compute analytical gradients for all trainable parameters.
        """
        probabilities, cache = self.forward(X)
        self._validate_y(y, X.shape[0])

        dlogits = softmax_cross_entropy_gradient(
            y,
            probabilities,
        )

        dW2 = cache["A1"].T @ dlogits
        db2 = np.sum(dlogits, axis=0)

        dA1 = dlogits @ self.W2.T
        dZ1 = dA1 * (cache["Z1"] > 0)

        dW1 = cache["X"].T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }

    def get_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """
        Return copies of trainable parameters.
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
        Replace trainable parameters after validating keys, shapes, and finiteness.
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
