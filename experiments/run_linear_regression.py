"""Train linear regression from scratch on synthetic data."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import make_linear_regression_data
from src.data.preprocessing import standardize_features, train_val_split
from src.models.linear_regression import LinearRegressionScratch
from src.optimization.gradient_descent import BatchGradientDescent
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


def train_linear_regression(
    model: LinearRegressionScratch,
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
    config_path = PROJECT_ROOT / "configs" / "linear_regression.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    training_config = config["training"]
    log_file = config["logging"]["log_file"]

    set_seed(seed)
    logger = get_logger("linear_regression_training", log_file=log_file)

    logger.info("Experiment: linear_regression_training")
    logger.info("Seed: %s", seed)

    X, y, true_weights, true_bias = make_linear_regression_data(
        n_samples=data_config["n_samples"],
        n_features=data_config["n_features"],
        noise=data_config["noise"],
        seed=seed,
    )

    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y,
        val_ratio=data_config["val_ratio"],
        seed=seed,
    )
    X_train_scaled, X_val_scaled, mean, std = standardize_features(X_train, X_val)

    model = LinearRegressionScratch(n_features=X_train_scaled.shape[1])
    optimizer = BatchGradientDescent(
        learning_rate=training_config["learning_rate"],
    )

    loss_history = train_linear_regression(
        model=model,
        optimizer=optimizer,
        X_train=X_train_scaled,
        y_train=y_train,
        num_epochs=training_config["num_epochs"],
        log_every=training_config["log_every"],
        logger=logger,
    )

    final_train_loss = model.compute_loss(X_train_scaled, y_train)
    final_val_loss = model.compute_loss(X_val_scaled, y_val)

    logger.info("Initial train loss: %.6f", loss_history[0])
    logger.info("Final train loss: %.6f", final_train_loss)
    logger.info("Final validation loss: %.6f", final_val_loss)
    logger.info("True weights: %s", true_weights)
    logger.info("Learned weights: %s", model.weights)
    logger.info("True bias: %.6f", true_bias)
    logger.info("Learned bias: %.6f", model.bias)
    logger.info("Number of epochs: %s", training_config["num_epochs"])
    logger.info("Learning rate: %s", training_config["learning_rate"])
    logger.info("Note: batch gradient descent on standardized synthetic data.")

    # Keep preprocessing statistics visible for beginner-friendly inspection.
    _ = mean, std


if __name__ == "__main__":
    main()
