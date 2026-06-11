"""Train a scratch MLP on nonlinear XOR-style synthetic data."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import make_xor_classification_data
from src.data.preprocessing import standardize_features, train_val_split
from src.models.mlp import BinaryMLPScratch
from src.optimization.parameter_optimizers import ParameterAdam
from src.train import evaluate_binary_mlp, train_binary_mlp
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_multiple_loss_curves
from src.utils.seed import set_seed


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "mlp.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    optimizer_config = config["optimizer"]
    loss_curve_path = config["output"]["loss_curve_path"]
    log_file = config["logging"]["log_file"]

    set_seed(seed)
    logger = get_logger("mlp_xor_training", log_file=log_file)

    logger.info("Experiment: mlp_xor_training")
    logger.info("Seed: %s", seed)

    X, y = make_xor_classification_data(
        n_samples=data_config["n_samples"],
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

    logger.info("Train shape: X=%s, y=%s", X_train_scaled.shape, y_train.shape)
    logger.info("Validation shape: X=%s, y=%s", X_val_scaled.shape, y_val.shape)
    logger.info("Train label counts: %s", np.bincount(y_train, minlength=2))
    logger.info("Validation label counts: %s", np.bincount(y_val, minlength=2))

    model = BinaryMLPScratch(
        n_features=X_train_scaled.shape[1],
        hidden_dim=model_config["hidden_dim"],
        seed=seed,
    )
    optimizer = ParameterAdam(
        learning_rate=optimizer_config["learning_rate"],
        beta1=optimizer_config["beta1"],
        beta2=optimizer_config["beta2"],
        epsilon=float(optimizer_config["epsilon"]),
    )

    threshold = training_config["threshold"]
    initial_train_metrics = evaluate_binary_mlp(
        model,
        X_train_scaled,
        y_train,
        threshold=threshold,
    )
    initial_val_metrics = evaluate_binary_mlp(
        model,
        X_val_scaled,
        y_val,
        threshold=threshold,
    )
    logger.info(
        "Initial train BCE: %.6f, accuracy: %.6f",
        initial_train_metrics["bce"],
        initial_train_metrics["accuracy"],
    )
    logger.info(
        "Initial validation BCE: %.6f, accuracy: %.6f",
        initial_val_metrics["bce"],
        initial_val_metrics["accuracy"],
    )

    history = train_binary_mlp(
        model=model,
        optimizer=optimizer,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        num_epochs=training_config["num_epochs"],
        batch_size=training_config["batch_size"],
        seed=seed,
        log_every=training_config["log_every"],
        logger=logger,
        threshold=threshold,
    )

    final_train_metrics = evaluate_binary_mlp(
        model,
        X_train_scaled,
        y_train,
        threshold=threshold,
    )
    final_val_metrics = evaluate_binary_mlp(
        model,
        X_val_scaled,
        y_val,
        threshold=threshold,
    )

    logger.info(
        "Final train BCE: %.6f, accuracy: %.6f",
        final_train_metrics["bce"],
        final_train_metrics["accuracy"],
    )
    logger.info(
        "Final validation BCE: %.6f, accuracy: %.6f",
        final_val_metrics["bce"],
        final_val_metrics["accuracy"],
    )
    logger.info("Number of parameter updates: %s", history["update_count"])
    logger.info("Number of epochs: %s", training_config["num_epochs"])
    logger.info("Batch size: %s", training_config["batch_size"])
    logger.info("Hidden dimension: %s", model_config["hidden_dim"])
    logger.info("Optimizer: %s", optimizer_config["name"])
    logger.info("Optimizer learning rate: %s", optimizer_config["learning_rate"])
    logger.info("Optimizer beta1: %s", optimizer_config["beta1"])
    logger.info("Optimizer beta2: %s", optimizer_config["beta2"])
    logger.info("Optimizer epsilon: %s", optimizer_config["epsilon"])

    plot_multiple_loss_curves(
        {
            "Train BCE": history["train_bce"],
            "Validation BCE": history["val_bce"],
        },
        output_path=loss_curve_path,
        title="MLP Training on Nonlinear XOR-Style Data",
        xlabel="Epoch",
        ylabel="Binary Cross Entropy Loss",
    )
    logger.info("Saved loss curve to: %s", loss_curve_path)

    # Keep preprocessing statistics visible for beginner-friendly inspection.
    _ = mean, std


if __name__ == "__main__":
    main()
