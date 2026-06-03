"""Compare optimizers on logistic regression with synthetic binary data."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import make_binary_classification_data
from src.data.preprocessing import (
    iterate_minibatches,
    standardize_features,
    train_val_split,
)
from src.evaluation.metrics import accuracy_score, binary_cross_entropy
from src.models.logistic_regression import LogisticRegressionScratch
from src.optimization.adam import Adam
from src.optimization.gradient_descent import BatchGradientDescent
from src.optimization.momentum import Momentum
from src.optimization.sgd import SGD
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_multiple_loss_curves
from src.utils.seed import set_seed


def train_full_batch(
    model,
    optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_epochs: int,
    log_every: int,
    logger,
    optimizer_name: str,
) -> tuple[list[float], int]:
    """Train with one full-batch parameter update per epoch."""
    loss_history = []
    update_count = 0

    for epoch in range(1, num_epochs + 1):
        dw, db = model.compute_gradients(X_train, y_train)
        model.weights, model.bias = optimizer.step(model.weights, model.bias, dw, db)
        update_count += 1

        train_probabilities = model.predict_proba(X_train)
        train_bce = binary_cross_entropy(y_train, train_probabilities)
        loss_history.append(train_bce)

        if epoch % log_every == 0:
            logger.info(
                "%s epoch %s/%s - train_bce: %.6f",
                optimizer_name,
                epoch,
                num_epochs,
                train_bce,
            )

    return loss_history, update_count


def train_minibatch(
    model,
    optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_epochs: int,
    batch_size: int,
    seed: int,
    log_every: int,
    logger,
    optimizer_name: str,
) -> tuple[list[float], int]:
    """Train with shuffled mini-batches and return epoch-level train BCE."""
    loss_history = []
    update_count = 0

    for epoch in range(1, num_epochs + 1):
        for X_batch, y_batch in iterate_minibatches(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            dw, db = model.compute_gradients(X_batch, y_batch)
            model.weights, model.bias = optimizer.step(
                model.weights,
                model.bias,
                dw,
                db,
            )
            update_count += 1

        train_probabilities = model.predict_proba(X_train)
        train_bce = binary_cross_entropy(y_train, train_probabilities)
        loss_history.append(train_bce)

        if epoch % log_every == 0:
            logger.info(
                "%s epoch %s/%s - train_bce: %.6f",
                optimizer_name,
                epoch,
                num_epochs,
                train_bce,
            )

    return loss_history, update_count


def evaluate_model(
    model: LogisticRegressionScratch,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate train BCE, validation BCE, and validation accuracy."""
    train_probabilities = model.predict_proba(X_train)
    val_probabilities = model.predict_proba(X_val)
    val_predictions = model.predict(X_val, threshold=threshold)

    return {
        "train_bce": binary_cross_entropy(y_train, train_probabilities),
        "val_bce": binary_cross_entropy(y_val, val_probabilities),
        "val_accuracy": accuracy_score(y_val, val_predictions),
    }


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "optimizer_comparison.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    training_config = config["training"]
    optimizer_config = config["optimizers"]
    log_file = config["logging"]["log_file"]
    loss_curve_path = config["output"]["loss_curve_path"]

    set_seed(seed)
    logger = get_logger("optimizer_comparison", log_file=log_file)

    logger.info("Experiment: optimizer_comparison")
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

    logger.info("Train shape: X=%s, y=%s", X_train_scaled.shape, y_train.shape)
    logger.info("Validation shape: X=%s, y=%s", X_val_scaled.shape, y_val.shape)
    logger.info("Every model starts from zero weights and zero bias.")
    logger.info(
        "This is a same-epoch teaching comparison. Mini-batch optimizers perform "
        "more parameter updates per epoch than full-batch gradient descent, so "
        "this is not a strict optimizer benchmark."
    )

    num_epochs = training_config["num_epochs"]
    batch_size = training_config["batch_size"]
    log_every = training_config["log_every"]
    threshold = training_config["threshold"]
    n_features = X_train_scaled.shape[1]

    histories = {}

    bgd_model = LogisticRegressionScratch(n_features=n_features)
    bgd_optimizer = BatchGradientDescent(
        learning_rate=optimizer_config["batch_gradient_descent"]["learning_rate"],
    )
    bgd_history, bgd_updates = train_full_batch(
        model=bgd_model,
        optimizer=bgd_optimizer,
        X_train=X_train_scaled,
        y_train=y_train,
        num_epochs=num_epochs,
        log_every=log_every,
        logger=logger,
        optimizer_name="Batch Gradient Descent",
    )
    histories["Batch Gradient Descent"] = bgd_history
    bgd_metrics = evaluate_model(
        bgd_model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        threshold,
    )

    sgd_model = LogisticRegressionScratch(n_features=n_features)
    sgd_optimizer = SGD(
        learning_rate=optimizer_config["sgd"]["learning_rate"],
    )
    sgd_history, sgd_updates = train_minibatch(
        model=sgd_model,
        optimizer=sgd_optimizer,
        X_train=X_train_scaled,
        y_train=y_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        log_every=log_every,
        logger=logger,
        optimizer_name="Mini-batch SGD",
    )
    histories["Mini-batch SGD"] = sgd_history
    sgd_metrics = evaluate_model(
        sgd_model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        threshold,
    )

    momentum_model = LogisticRegressionScratch(n_features=n_features)
    momentum_optimizer = Momentum(
        learning_rate=optimizer_config["momentum"]["learning_rate"],
        beta=optimizer_config["momentum"]["beta"],
    )
    momentum_history, momentum_updates = train_minibatch(
        model=momentum_model,
        optimizer=momentum_optimizer,
        X_train=X_train_scaled,
        y_train=y_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        log_every=log_every,
        logger=logger,
        optimizer_name="Momentum",
    )
    histories["Momentum"] = momentum_history
    momentum_metrics = evaluate_model(
        momentum_model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        threshold,
    )

    adam_model = LogisticRegressionScratch(n_features=n_features)
    adam_optimizer = Adam(
        learning_rate=optimizer_config["adam"]["learning_rate"],
        beta1=optimizer_config["adam"]["beta1"],
        beta2=optimizer_config["adam"]["beta2"],
        epsilon=float(optimizer_config["adam"]["epsilon"]),
    )
    adam_history, adam_updates = train_minibatch(
        model=adam_model,
        optimizer=adam_optimizer,
        X_train=X_train_scaled,
        y_train=y_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        log_every=log_every,
        logger=logger,
        optimizer_name="Adam",
    )
    histories["Adam"] = adam_history
    adam_metrics = evaluate_model(
        adam_model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        threshold,
    )

    results = [
        ("Batch Gradient Descent", bgd_updates, bgd_metrics),
        ("Mini-batch SGD", sgd_updates, sgd_metrics),
        ("Momentum", momentum_updates, momentum_metrics),
        ("Adam", adam_updates, adam_metrics),
    ]
    for optimizer_name, update_count, metrics in results:
        logger.info("Optimizer: %s", optimizer_name)
        logger.info("Number of epochs: %s", num_epochs)
        logger.info("Number of parameter updates: %s", update_count)
        logger.info("Final train BCE: %.6f", metrics["train_bce"])
        logger.info("Final validation BCE: %.6f", metrics["val_bce"])
        logger.info("Validation accuracy: %.6f", metrics["val_accuracy"])

    plot_multiple_loss_curves(
        histories,
        output_path=loss_curve_path,
        title="Optimizer Comparison on Logistic Regression",
        xlabel="Epoch",
        ylabel="Binary Cross Entropy Loss",
    )
    logger.info("Saved loss curve to: %s", loss_curve_path)

    # Keep preprocessing statistics visible for beginner-friendly inspection.
    _ = mean, std


if __name__ == "__main__":
    main()
