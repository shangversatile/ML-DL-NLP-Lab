"""Controlled overfitting analysis for scratch MLP training."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import make_xor_classification_data
from src.data.preprocessing import (
    flip_binary_labels,
    standardize_features,
    train_val_split,
)
from src.models.mlp import BinaryMLPScratch
from src.optimization.parameter_optimizers import ParameterAdam
from src.train import evaluate_binary_mlp, train_binary_mlp
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_multiple_loss_curves
from src.utils.seed import set_seed


def create_adam(optimizer_config: dict) -> ParameterAdam:
    return ParameterAdam(
        learning_rate=optimizer_config["learning_rate"],
        beta1=optimizer_config["beta1"],
        beta2=optimizer_config["beta2"],
        epsilon=float(optimizer_config["epsilon"]),
    )


def summarize_history(
    history: dict,
) -> dict[str, float | int]:
    """
    Summarize validation-BCE dynamics.
    """
    val_bce = history["val_bce"]
    best_index = int(np.argmin(val_bce))
    best_epoch = best_index + 1

    best_val_bce = float(val_bce[best_index])
    final_val_bce = float(val_bce[-1])
    final_train_bce = float(history["train_bce"][-1])
    final_train_accuracy = float(history["train_accuracy"][-1])
    final_val_accuracy = float(history["val_accuracy"][-1])
    post_best_val_bce_increase = final_val_bce - best_val_bce

    return {
        "best_epoch": best_epoch,
        "best_val_bce": best_val_bce,
        "final_val_bce": final_val_bce,
        "final_train_bce": final_train_bce,
        "final_train_accuracy": final_train_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "post_best_val_bce_increase": post_best_val_bce_increase,
    }


def run_condition(
    condition_name: str,
    training_labels: np.ndarray,
    clean_training_labels: np.ndarray,
    initial_parameters: dict[str, np.ndarray],
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int,
    seed: int,
    num_epochs: int,
    batch_size: int,
    log_every: int,
    threshold: float,
    optimizer_config: dict,
    logger,
) -> dict:
    """
    Train one model under one label condition.
    """
    model = BinaryMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=hidden_dim,
        seed=seed,
    )
    model.set_parameters(initial_parameters)

    optimizer = create_adam(optimizer_config)

    initial_validation_metrics = evaluate_binary_mlp(
        model,
        X_val,
        y_val,
        threshold=threshold,
    )

    history = train_binary_mlp(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=training_labels,
        X_val=X_val,
        y_val=y_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        log_every=log_every,
        logger=logger,
        threshold=threshold,
    )

    final_validation_metrics = evaluate_binary_mlp(
        model,
        X_val,
        y_val,
        threshold=threshold,
    )
    final_training_metrics_against_objective = evaluate_binary_mlp(
        model,
        X_train,
        training_labels,
        threshold=threshold,
    )
    final_training_metrics_against_clean_labels = evaluate_binary_mlp(
        model,
        X_train,
        clean_training_labels,
        threshold=threshold,
    )
    summary = summarize_history(history)

    logger.info("Condition: %s", condition_name)
    logger.info(
        "Initial validation BCE: %.6f",
        initial_validation_metrics["bce"],
    )
    logger.info("Best validation BCE: %.6f", summary["best_val_bce"])
    logger.info("Best validation epoch: %s", summary["best_epoch"])
    logger.info(
        "Final train BCE against training objective: %.6f",
        final_training_metrics_against_objective["bce"],
    )
    logger.info("Final validation BCE: %.6f", final_validation_metrics["bce"])
    logger.info(
        "Post-best validation BCE increase: %.6f",
        summary["post_best_val_bce_increase"],
    )
    logger.info(
        "Final train accuracy against condition-specific labels: %.6f",
        final_training_metrics_against_objective["accuracy"],
    )
    logger.info(
        "Final train accuracy against clean labels: %.6f",
        final_training_metrics_against_clean_labels["accuracy"],
    )
    logger.info(
        "Final validation accuracy: %.6f",
        final_validation_metrics["accuracy"],
    )
    logger.info("Update count: %s", history["update_count"])

    return {
        "condition_name": condition_name,
        "history": history,
        "summary": summary,
        "final_validation_metrics": final_validation_metrics,
        "final_training_metrics_against_objective": (
            final_training_metrics_against_objective
        ),
        "final_training_metrics_against_clean_labels": (
            final_training_metrics_against_clean_labels
        ),
        "final_parameters": model.get_parameters(),
    }


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "mlp_overfitting_analysis.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    optimizer_config = config["optimizer"]
    output_config = config["output"]
    log_file = config["logging"]["log_file"]

    set_seed(seed)
    logger = get_logger("mlp_controlled_overfitting", log_file=log_file)

    logger.info("Experiment: mlp_controlled_overfitting")
    logger.info("Seed: %s", seed)

    X, y = make_xor_classification_data(
        n_samples=data_config["n_samples"],
        noise=data_config["feature_noise"],
        seed=seed,
    )
    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y,
        val_ratio=data_config["val_ratio"],
        seed=seed,
    )
    X_train_scaled, X_val_scaled, mean, std = standardize_features(X_train, X_val)

    y_train_clean = y_train.copy()
    y_train_corrupted = flip_binary_labels(
        y_train_clean,
        flip_rate=data_config["train_label_flip_rate"],
        seed=seed + 1,
    )
    n_flipped = int(np.sum(y_train_clean != y_train_corrupted))

    logger.info("Train shape: X=%s, y=%s", X_train_scaled.shape, y_train.shape)
    logger.info("Validation shape: X=%s, y=%s", X_val_scaled.shape, y_val.shape)
    logger.info("Clean train label counts: %s", np.bincount(y_train_clean, minlength=2))
    logger.info(
        "Corrupted train label counts: %s",
        np.bincount(y_train_corrupted, minlength=2),
    )
    logger.info("Validation label counts: %s", np.bincount(y_val, minlength=2))
    logger.info("Training labels flipped: %s", n_flipped)

    base_model = BinaryMLPScratch(
        n_features=X_train_scaled.shape[1],
        hidden_dim=model_config["hidden_dim"],
        seed=seed,
    )
    initial_parameters = base_model.get_parameters()

    common_arguments = {
        "clean_training_labels": y_train_clean,
        "initial_parameters": initial_parameters,
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "hidden_dim": model_config["hidden_dim"],
        "seed": seed,
        "num_epochs": training_config["num_epochs"],
        "batch_size": training_config["batch_size"],
        "log_every": training_config["log_every"],
        "threshold": training_config["threshold"],
        "optimizer_config": optimizer_config,
        "logger": logger,
    }

    clean_result = run_condition(
        condition_name="clean_labels",
        training_labels=y_train_clean,
        **common_arguments,
    )
    corrupted_result = run_condition(
        condition_name="corrupted_labels",
        training_labels=y_train_corrupted,
        **common_arguments,
    )

    update_counts = {
        clean_result["condition_name"]: clean_result["history"]["update_count"],
        corrupted_result["condition_name"]: corrupted_result["history"]["update_count"],
    }
    if len(set(update_counts.values())) != 1:
        raise ValueError(f"Condition update counts differ: {update_counts}")
    logger.info("Both condition update counts are equal: %s", update_counts)

    plot_multiple_loss_curves(
        {
            "Clean condition: train BCE": clean_result["history"]["train_bce"],
            "Clean condition: validation BCE": clean_result["history"]["val_bce"],
            "Corrupted condition: train BCE": corrupted_result["history"]["train_bce"],
            "Corrupted condition: validation BCE": (
                corrupted_result["history"]["val_bce"]
            ),
        },
        output_path=output_config["bce_curve_path"],
        title="Controlled Overfitting Analysis: BCE",
        xlabel="Epoch",
        ylabel="Binary Cross Entropy Loss",
    )
    plot_multiple_loss_curves(
        {
            "Clean condition: train accuracy": (
                clean_result["history"]["train_accuracy"]
            ),
            "Clean condition: validation accuracy": (
                clean_result["history"]["val_accuracy"]
            ),
            "Corrupted condition: train-objective accuracy": (
                corrupted_result["history"]["train_accuracy"]
            ),
            "Corrupted condition: validation accuracy": (
                corrupted_result["history"]["val_accuracy"]
            ),
        },
        output_path=output_config["accuracy_curve_path"],
        title="Controlled Overfitting Analysis: Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
    )
    logger.info("Saved BCE curve to: %s", output_config["bce_curve_path"])
    logger.info("Saved accuracy curve to: %s", output_config["accuracy_curve_path"])
    logger.info(
        "This experiment is designed to expose a controlled overfitting signal."
    )
    logger.info(
        "The main comparison is between clean-label and corrupted-label training."
    )
    logger.info(
        (
            "A validation-BCE increase after the best epoch is a diagnostic signal, "
            "not a universal threshold or proof of one unique causal mechanism."
        )
    )

    # Keep preprocessing statistics visible for beginner-friendly inspection.
    _ = mean, std


if __name__ == "__main__":
    main()
