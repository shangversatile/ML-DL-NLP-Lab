"""Reusable training utilities."""

import numpy as np

from src.data.preprocessing import iterate_minibatches
from src.evaluation.metrics import accuracy_score, binary_cross_entropy
from src.models.mlp import BinaryMLPScratch


def _validate_positive_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")


def map_mlp_gradients_to_parameters(
    gradients: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Convert MLP backpropagation gradient names into optimizer parameter keys.
    """
    expected_keys = {"dW1", "db1", "dW2", "db2"}
    if set(gradients) != expected_keys:
        raise ValueError("MLP gradients must contain dW1, db1, dW2, and db2.")

    # The training layer performs semantic mapping from model-gradient names
    # to parameter names. Optimizers only receive generic matching keys.
    return {
        "W1": gradients["dW1"],
        "b1": gradients["db1"],
        "W2": gradients["dW2"],
        "b2": gradients["db2"],
    }


def evaluate_binary_mlp(
    model: BinaryMLPScratch,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute BCE and accuracy for a binary MLP.
    """
    probabilities = model.predict_proba(X)
    predictions = model.predict(X, threshold=threshold)

    return {
        "bce": binary_cross_entropy(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
    }


def train_binary_mlp(
    model: BinaryMLPScratch,
    optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int,
    batch_size: int,
    seed: int,
    log_every: int,
    logger,
    threshold: float = 0.5,
) -> dict[str, list[float] | int]:
    """
    Train a binary MLP with reproducible shuffled mini-batches.
    """
    _validate_positive_int("num_epochs", num_epochs)
    _validate_positive_int("batch_size", batch_size)
    _validate_int("seed", seed)
    _validate_positive_int("log_every", log_every)

    train_bce_history = []
    val_bce_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    update_count = 0

    for epoch in range(1, num_epochs + 1):
        for X_batch, y_batch in iterate_minibatches(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            raw_gradients = model.compute_gradients(X_batch, y_batch)
            parameter_gradients = map_mlp_gradients_to_parameters(raw_gradients)
            parameters = model.get_parameters()
            updated_parameters = optimizer.step(
                parameters,
                parameter_gradients,
            )
            model.set_parameters(updated_parameters)
            update_count += 1

        train_metrics = evaluate_binary_mlp(
            model,
            X_train,
            y_train,
            threshold=threshold,
        )
        val_metrics = evaluate_binary_mlp(
            model,
            X_val,
            y_val,
            threshold=threshold,
        )

        train_bce_history.append(train_metrics["bce"])
        val_bce_history.append(val_metrics["bce"])
        train_accuracy_history.append(train_metrics["accuracy"])
        val_accuracy_history.append(val_metrics["accuracy"])

        if epoch % log_every == 0:
            logger.info(
                (
                    "MLP epoch %s/%s - train_bce: %.6f - "
                    "val_bce: %.6f - train_accuracy: %.6f - "
                    "val_accuracy: %.6f"
                ),
                epoch,
                num_epochs,
                train_metrics["bce"],
                val_metrics["bce"],
                train_metrics["accuracy"],
                val_metrics["accuracy"],
            )

    return {
        "train_bce": train_bce_history,
        "val_bce": val_bce_history,
        "train_accuracy": train_accuracy_history,
        "val_accuracy": val_accuracy_history,
        "update_count": update_count,
    }
