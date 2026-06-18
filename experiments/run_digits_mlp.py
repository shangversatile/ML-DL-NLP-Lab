"""Train a scratch multiclass MLP on real handwritten digit data."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.digits import load_digits_dataset
from src.data.preprocessing import stratified_train_val_test_split
from src.models.multiclass_mlp import MulticlassMLPScratch
from src.optimization.parameter_optimizers import ParameterAdam
from src.train import evaluate_multiclass_mlp, train_multiclass_mlp
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_multiple_loss_curves
from src.utils.seed import set_seed


def summarize_class_counts(
    y: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    return np.bincount(y, minlength=num_classes)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "digits_mlp.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    optimizer_config = config["optimizer"]
    output_config = config["output"]
    log_file = config["logging"]["log_file"]

    set_seed(seed)
    logger = get_logger("digits_mlp_training", log_file=log_file)

    logger.info("Experiment: digits_mlp_training")
    logger.info("Seed: %s", seed)

    X, y, images = load_digits_dataset(
        scale_pixels=data_config["scale_pixels"],
    )
    X_train, X_val, X_test, y_train, y_val, y_test = (
        stratified_train_val_test_split(
            X,
            y,
            val_ratio=data_config["val_ratio"],
            test_ratio=data_config["test_ratio"],
            seed=seed,
        )
    )

    logger.info("Full X shape: %s", X.shape)
    logger.info("Images shape: %s", images.shape)
    logger.info("Train shape: X=%s, y=%s", X_train.shape, y_train.shape)
    logger.info("Validation shape: X=%s, y=%s", X_val.shape, y_val.shape)
    logger.info("Test shape: X=%s, y=%s", X_test.shape, y_test.shape)
    logger.info(
        "Train class counts: %s",
        summarize_class_counts(y_train, model_config["num_classes"]),
    )
    logger.info(
        "Validation class counts: %s",
        summarize_class_counts(y_val, model_config["num_classes"]),
    )
    logger.info(
        "Test class counts: %s",
        summarize_class_counts(y_test, model_config["num_classes"]),
    )
    logger.info("Scale pixels to [0, 1]: %s", data_config["scale_pixels"])

    model = MulticlassMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        seed=seed,
    )

    if optimizer_config["name"].lower() != "adam":
        raise ValueError("Only the Adam optimizer is configured for this experiment.")

    optimizer = ParameterAdam(
        learning_rate=optimizer_config["learning_rate"],
        beta1=optimizer_config["beta1"],
        beta2=optimizer_config["beta2"],
        epsilon=float(optimizer_config["epsilon"]),
    )

    initial_train_metrics = evaluate_multiclass_mlp(model, X_train, y_train)
    initial_val_metrics = evaluate_multiclass_mlp(model, X_val, y_val)
    initial_test_metrics = evaluate_multiclass_mlp(model, X_test, y_test)

    logger.info(
        "Initial train CE: %.6f, accuracy: %.6f",
        initial_train_metrics["cross_entropy"],
        initial_train_metrics["accuracy"],
    )
    logger.info(
        "Initial validation CE: %.6f, accuracy: %.6f",
        initial_val_metrics["cross_entropy"],
        initial_val_metrics["accuracy"],
    )
    logger.info(
        "Initial test CE: %.6f, accuracy: %.6f",
        initial_test_metrics["cross_entropy"],
        initial_test_metrics["accuracy"],
    )

    history = train_multiclass_mlp(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=training_config["num_epochs"],
        batch_size=training_config["batch_size"],
        seed=seed,
        log_every=training_config["log_every"],
        logger=logger,
    )

    final_train_metrics = evaluate_multiclass_mlp(model, X_train, y_train)
    final_val_metrics = evaluate_multiclass_mlp(model, X_val, y_val)
    final_test_metrics = evaluate_multiclass_mlp(model, X_test, y_test)

    logger.info(
        "Final train CE: %.6f, accuracy: %.6f",
        final_train_metrics["cross_entropy"],
        final_train_metrics["accuracy"],
    )
    logger.info(
        "Final validation CE: %.6f, accuracy: %.6f",
        final_val_metrics["cross_entropy"],
        final_val_metrics["accuracy"],
    )
    logger.info(
        "Final test CE: %.6f, accuracy: %.6f",
        final_test_metrics["cross_entropy"],
        final_test_metrics["accuracy"],
    )
    logger.info("Number of parameter updates: %s", history["update_count"])
    logger.info("Hidden dimension: %s", model_config["hidden_dim"])
    logger.info("Optimizer: %s", optimizer_config["name"])
    logger.info("Optimizer learning rate: %s", optimizer_config["learning_rate"])
    logger.info("Optimizer beta1: %s", optimizer_config["beta1"])
    logger.info("Optimizer beta2: %s", optimizer_config["beta2"])
    logger.info("Optimizer epsilon: %s", optimizer_config["epsilon"])

    plot_multiple_loss_curves(
        {
            "Train CE": history["train_cross_entropy"],
            "Validation CE": history["val_cross_entropy"],
        },
        output_path=output_config["loss_curve_path"],
        title="Digits MLP Training: Cross Entropy",
        xlabel="Epoch",
        ylabel="Cross Entropy Loss",
    )
    plot_multiple_loss_curves(
        {
            "Train Accuracy": history["train_accuracy"],
            "Validation Accuracy": history["val_accuracy"],
        },
        output_path=output_config["accuracy_curve_path"],
        title="Digits MLP Training: Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
    )
    logger.info("Saved cross entropy curve to: %s", output_config["loss_curve_path"])
    logger.info("Saved accuracy curve to: %s", output_config["accuracy_curve_path"])
    logger.info(
        (
            "This is a baseline real-data training run for the scratch multiclass "
            "MLP. It does not include checkpointing, confusion matrix analysis, "
            "calibration, or local handwritten-input distribution-shift analysis yet."
        )
    )


if __name__ == "__main__":
    main()
