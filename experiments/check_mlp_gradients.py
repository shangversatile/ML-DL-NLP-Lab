"""Run numerical gradient checking for the scratch MLP."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mlp import BinaryMLPScratch
from src.utils.gradient_check import compare_gradients, compute_numerical_gradients


def create_gradient_check_case():
    model = BinaryMLPScratch(n_features=2, hidden_dim=2, seed=42)

    model.W1 = np.array(
        [
            [1.0, -1.0],
            [0.5, 2.0],
        ]
    )
    model.b1 = np.array([0.1, -0.2])

    model.W2 = np.array(
        [
            [0.7],
            [-1.2],
        ]
    )
    model.b2 = np.array([0.05])

    X = np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [2.0, -0.5],
        ]
    )
    y = np.array([1, 0, 1])

    return model, X, y


def main() -> None:
    model, X, y = create_gradient_check_case()

    analytical_gradients = model.compute_gradients(X, y)
    numerical_gradients = compute_numerical_gradients(model, X, y)
    errors = compare_gradients(analytical_gradients, numerical_gradients)

    tolerance = 1e-6
    passed = all(error < tolerance for error in errors.values())

    print("MLP numerical gradient check")
    for parameter_name in ("W1", "b1", "W2", "b2"):
        print(
            f"Parameter: {parameter_name} - "
            f"relative L2 error: {errors[parameter_name]:.12e}"
        )
    print(f"Gradient check passed: {passed}")
    print("Note: this check uses central finite differences on a small deterministic case.")
    print(
        "Note: hidden pre-activations are intentionally kept away from zero "
        "to avoid ReLU nondifferentiability."
    )


if __name__ == "__main__":
    main()
