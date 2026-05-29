"""Train logistic regression from scratch on synthetic binary data."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import make_binary_classification_data
from src.data.preprocessing import standardize_features, train_val_split
from src.evaluation.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.models.logistic_regression import LogisticRegressionScratch
from src.optimization.gradient_descent import BatchGradientDescent
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


def train_logistic_regression(
    model: LogisticRegressionScratch,
    optimizer: BatchGradientDescent,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_epochs: int,
    log_every: int,
    logger,
) -> list[float]:
    """Run batch gradient descent and return the train loss history."""
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = model.compute_loss(X_train, y_train)
        dw, db = model.compute_gradients(X_train, y_train)
        model.weights, model.bias = optimizer.step(model.weights, model.bias, dw, db)

        loss_history.append(train_loss)

        if epoch % log_every == 0:
            logger.info("Epoch %s/%s - train_loss: %.6f", epoch, num_epochs, train_loss)

    return loss_history


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "logistic_regression.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    training_config = config["training"]
    log_file = config["logging"]["log_file"]

    set_seed(seed)
    logger = get_logger("logistic_regression_training", log_file=log_file)

    logger.info("Experiment: logistic_regression_training")
    logger.info("Seed: %s", seed)

    X, y = make_binary_classification_data(
        n_samples=data_config["n_samples"],
        n_features=data_config["n_features"],
        seed=seed,
    )

    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y,
        val_ratio=data_config["val_ratio"],
        seed=seed,
    )
    X_train_scaled, X_val_scaled, mean, std = standardize_features(X_train, X_val)

    model = LogisticRegressionScratch(n_features=X_train_scaled.shape[1])
    optimizer = BatchGradientDescent(
        learning_rate=training_config["learning_rate"],
    )

    loss_history = train_logistic_regression(
        model=model,
        optimizer=optimizer,
        X_train=X_train_scaled,
        y_train=y_train,
        num_epochs=training_config["num_epochs"],
        log_every=training_config["log_every"],
        logger=logger,
    )

    threshold = training_config["threshold"]
    final_train_loss = model.compute_loss(X_train_scaled, y_train)
    final_val_loss = model.compute_loss(X_val_scaled, y_val)

    train_probabilities = model.predict_proba(X_train_scaled)
    val_probabilities = model.predict_proba(X_val_scaled)
    train_predictions = model.predict(X_train_scaled, threshold=threshold)
    val_predictions = model.predict(X_val_scaled, threshold=threshold)

    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions)
    train_recall = recall_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions)
    val_recall = recall_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions)
    val_confusion_matrix = confusion_matrix(y_val, val_predictions)

    logger.info("Initial train loss: %.6f", loss_history[0])
    logger.info("Final train loss: %.6f", final_train_loss)
    logger.info("Final validation loss: %.6f", final_val_loss)
    logger.info(
        "Train metrics - accuracy: %.6f, precision: %.6f, recall: %.6f, f1: %.6f",
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
    )
    logger.info(
        "Validation metrics - accuracy: %.6f, precision: %.6f, recall: %.6f, f1: %.6f",
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
    )
    logger.info("Validation confusion matrix:\n%s", val_confusion_matrix)
    logger.info("Learned weights in standardized space: %s", model.weights)
    logger.info("Learned bias in standardized space: %.6f", model.bias)
    logger.info("Number of epochs: %s", training_config["num_epochs"])
    logger.info("Learning rate: %s", training_config["learning_rate"])
    logger.info("Threshold: %s", threshold)
    logger.info("Note: logistic regression trained with batch gradient descent.")

    # Keep intermediate values visible for beginner-friendly inspection.
    _ = mean, std, train_probabilities, val_probabilities


if __name__ == "__main__":
    main()
