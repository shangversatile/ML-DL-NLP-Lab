import logging

import numpy as np
import pytest

from src.data.datasets import make_xor_classification_data
from src.data.preprocessing import standardize_features, train_val_split
from src.models.mlp import BinaryMLPScratch
from src.optimization.parameter_optimizers import ParameterAdam
from src.train import (
    evaluate_binary_mlp,
    map_mlp_gradients_to_parameters,
    train_binary_mlp,
)


logger = logging.getLogger("test_mlp_training")


def test_map_mlp_gradients_to_parameters() -> None:
    gradients = {
        "dW1": np.array([[1.0, 2.0]]),
        "db1": np.array([3.0]),
        "dW2": np.array([[4.0], [5.0]]),
        "db2": np.array([6.0]),
    }

    mapped_gradients = map_mlp_gradients_to_parameters(gradients)

    assert set(mapped_gradients) == {"W1", "b1", "W2", "b2"}
    assert np.array_equal(mapped_gradients["W1"], gradients["dW1"])
    assert np.array_equal(mapped_gradients["b1"], gradients["db1"])
    assert np.array_equal(mapped_gradients["W2"], gradients["dW2"])
    assert np.array_equal(mapped_gradients["b2"], gradients["db2"])


def test_map_mlp_gradients_to_parameters_invalid_keys() -> None:
    gradients = {
        "dW1": np.array([[1.0, 2.0]]),
        "db1": np.array([3.0]),
        "dW2": np.array([[4.0], [5.0]]),
    }

    with pytest.raises(ValueError):
        map_mlp_gradients_to_parameters(gradients)


def _create_training_case(
    n_samples: int,
    noise: float,
    seed: int,
    val_ratio: float,
    hidden_dim: int,
) -> tuple[BinaryMLPScratch, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_xor_classification_data(
        n_samples=n_samples,
        noise=noise,
        seed=seed,
    )
    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y,
        val_ratio=val_ratio,
        seed=seed,
    )
    X_train_scaled, X_val_scaled, _, _ = standardize_features(X_train, X_val)
    model = BinaryMLPScratch(
        n_features=X_train_scaled.shape[1],
        hidden_dim=hidden_dim,
        seed=seed,
    )

    return model, X_train_scaled, X_val_scaled, y_train, y_val


def test_train_binary_mlp_reduces_bce_on_xor_data() -> None:
    seed = 42
    num_epochs = 200
    batch_size = 32
    model, X_train, X_val, y_train, y_val = _create_training_case(
        n_samples=300,
        noise=0.10,
        seed=seed,
        val_ratio=0.2,
        hidden_dim=12,
    )
    optimizer = ParameterAdam(learning_rate=0.01)

    initial_train_metrics = evaluate_binary_mlp(model, X_train, y_train)
    initial_val_metrics = evaluate_binary_mlp(model, X_val, y_val)

    history = train_binary_mlp(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        log_every=100,
        logger=logger,
    )

    final_train_metrics = evaluate_binary_mlp(model, X_train, y_train)
    final_val_metrics = evaluate_binary_mlp(model, X_val, y_val)
    expected_updates = num_epochs * int(np.ceil(len(X_train) / batch_size))

    assert final_train_metrics["bce"] < initial_train_metrics["bce"]
    assert final_val_metrics["bce"] < initial_val_metrics["bce"]
    assert final_val_metrics["accuracy"] > 0.75
    assert history["update_count"] == expected_updates
    assert len(history["train_bce"]) == num_epochs
    assert len(history["val_bce"]) == num_epochs
    assert len(history["train_accuracy"]) == num_epochs
    assert len(history["val_accuracy"]) == num_epochs


def test_train_binary_mlp_changes_parameters() -> None:
    seed = 42
    model, X_train, X_val, y_train, y_val = _create_training_case(
        n_samples=120,
        noise=0.10,
        seed=seed,
        val_ratio=0.2,
        hidden_dim=8,
    )
    optimizer = ParameterAdam(learning_rate=0.01)
    initial_parameters = model.get_parameters()

    train_binary_mlp(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=10,
        batch_size=16,
        seed=seed,
        log_every=10,
        logger=logger,
    )

    updated_parameters = model.get_parameters()

    assert any(
        not np.array_equal(initial_parameters[name], updated_parameters[name])
        for name in initial_parameters
    )
