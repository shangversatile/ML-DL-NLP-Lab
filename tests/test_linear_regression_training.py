"""Integration tests for linear regression training."""

from src.data.datasets import make_linear_regression_data
from src.data.preprocessing import standardize_features, train_val_split
from src.models.linear_regression import LinearRegressionScratch
from src.optimization.gradient_descent import BatchGradientDescent


def test_linear_regression_training_reduces_loss() -> None:
    X, y, _, _ = make_linear_regression_data(
        n_samples=100,
        n_features=2,
        noise=0.1,
        seed=42,
    )
    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y,
        val_ratio=0.2,
        seed=42,
    )
    X_train_scaled, X_val_scaled, _, _ = standardize_features(X_train, X_val)

    model = LinearRegressionScratch(n_features=2)
    optimizer = BatchGradientDescent(learning_rate=0.05)

    initial_loss = model.compute_loss(X_train_scaled, y_train)

    for _ in range(100):
        dw, db = model.compute_gradients(X_train_scaled, y_train)
        model.weights, model.bias = optimizer.step(
            model.weights,
            model.bias,
            dw,
            db,
        )

    final_loss = model.compute_loss(X_train_scaled, y_train)
    val_loss = model.compute_loss(X_val_scaled, y_val)

    assert final_loss < initial_loss
    assert final_loss < 0.05
    assert val_loss < 0.1
