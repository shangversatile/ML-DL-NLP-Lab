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
    binary_cross_entropy,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.models.logistic_regression import LogisticRegressionScratch
from src.optimization.gradient_descent import BatchGradientDescent
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_loss_curve
from src.utils.seed import set_seed


def summarize_binary_labels(y: np.ndarray) -> dict[str, float]:
    """Return simple class distribution statistics for binary labels."""
    num_samples = y.shape[0]
    num_negative = int(np.sum(y == 0))
    num_positive = int(np.sum(y == 1))

    return {
        "num_samples": num_samples,
        "num_negative": num_negative,
        "num_positive": num_positive,
        "negative_rate": num_negative / num_samples,
        "positive_rate": num_positive / num_samples,
    }


def evaluate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    """Return common binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


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
    evaluation_config = config["evaluation"]
    log_file = config["logging"]["log_file"]
    loss_curve_path = config["output"]["loss_curve_path"]

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

    train_label_distribution = summarize_binary_labels(y_train)
    val_label_distribution = summarize_binary_labels(y_val)
    logger.info("Train label distribution: %s", train_label_distribution)
    logger.info("Validation label distribution: %s", val_label_distribution)

    if (
        train_label_distribution["num_negative"] == 0
        or train_label_distribution["num_positive"] == 0
    ):
        logger.warning(
            "WARNING: train set contains only one class; "
            "classification metrics may be misleading."
        )
    if (
        val_label_distribution["num_negative"] == 0
        or val_label_distribution["num_positive"] == 0
    ):
        logger.warning(
            "WARNING: validation set contains only one class; "
            "classification metrics may be misleading."
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
    train_bce = binary_cross_entropy(y_train, train_probabilities)
    val_bce = binary_cross_entropy(y_val, val_probabilities)
    train_predictions = model.predict(X_train_scaled, threshold=threshold)
    val_predictions = model.predict(X_val_scaled, threshold=threshold)

    train_metrics = evaluate_classification_metrics(y_train, train_predictions)
    val_metrics = evaluate_classification_metrics(y_val, val_predictions)

    logger.info("Initial train loss: %.6f", loss_history[0])
    logger.info("Final train loss: %.6f", final_train_loss)
    logger.info("Final validation loss: %.6f", final_val_loss)
    logger.info("Evaluation train BCE: %.6f", train_bce)
    logger.info("Evaluation validation BCE: %.6f", val_bce)
    logger.info(
        "Train metrics - accuracy: %.6f, precision: %.6f, recall: %.6f, f1: %.6f",
        train_metrics["accuracy"],
        train_metrics["precision"],
        train_metrics["recall"],
        train_metrics["f1"],
    )
    logger.info(
        "Validation metrics - accuracy: %.6f, precision: %.6f, recall: %.6f, f1: %.6f",
        val_metrics["accuracy"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
    )
    logger.info("Validation confusion matrix:\n%s", val_metrics["confusion_matrix"])
    logger.info("Learned weights in standardized space: %s", model.weights)
    logger.info("Learned bias in standardized space: %.6f", model.bias)
    logger.info("Number of epochs: %s", training_config["num_epochs"])
    logger.info("Learning rate: %s", training_config["learning_rate"])
    logger.info("Threshold: %s", threshold)
    logger.info("Note: logistic regression trained with batch gradient descent.")
    logger.info("Threshold analysis:")
    for threshold_value in evaluation_config["thresholds"]:
        val_predictions_at_threshold = model.predict(
            X_val_scaled,
            threshold=threshold_value,
        )
        threshold_metrics = evaluate_classification_metrics(
            y_val,
            val_predictions_at_threshold,
        )
        logger.info(
            "threshold=%s | accuracy=%.6f precision=%.6f recall=%.6f f1=%.6f",
            threshold_value,
            threshold_metrics["accuracy"],
            threshold_metrics["precision"],
            threshold_metrics["recall"],
            threshold_metrics["f1"],
        )
        logger.info("confusion matrix:\n%s", threshold_metrics["confusion_matrix"])

    plot_loss_curve(
        loss_history,
        output_path=loss_curve_path,
        title="Logistic Regression Training Loss",
        xlabel="Epoch",
        ylabel="Binary Cross Entropy Loss",
    )
    logger.info("Saved loss curve to: %s", loss_curve_path)

    # Keep intermediate values visible for beginner-friendly inspection.
    _ = mean, std, train_probabilities, val_probabilities


if __name__ == "__main__":
    main()
