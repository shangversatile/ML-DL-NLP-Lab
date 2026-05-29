"""Integration test for logistic regression training."""

import numpy as np

from src.data.datasets import make_binary_classification_data
from src.data.preprocessing import standardize_features, train_val_split
from src.evaluation.metrics import accuracy_score, f1_score
from src.models.logistic_regression import LogisticRegressionScratch
from src.optimization.gradient_descent import BatchGradientDescent


def test_logistic_regression_training_reduces_loss_and_learns_boundary():
    X, y = make_binary_classification_data(
        n_samples=200,
        n_features=2,
        seed=42,
    )
    assert set(np.unique(y)) == {0, 1}

    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y,
        val_ratio=0.2,
        seed=42,
    )
    X_train_scaled, X_val_scaled, _, _ = standardize_features(X_train, X_val)

    model = LogisticRegressionScratch(n_features=2)
    optimizer = BatchGradientDescent(learning_rate=0.1)

    initial_loss = model.compute_loss(X_train_scaled, y_train)

    for _ in range(300):
        dw, db = model.compute_gradients(X_train_scaled, y_train)
        model.weights, model.bias = optimizer.step(
            model.weights,
            model.bias,
            dw,
            db,
        )

    final_loss = model.compute_loss(X_train_scaled, y_train)
    val_predictions = model.predict(X_val_scaled, threshold=0.5)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions)

    assert final_loss < initial_loss
    assert final_loss < 0.2
    assert val_accuracy >= 0.85
    assert val_f1 >= 0.85
