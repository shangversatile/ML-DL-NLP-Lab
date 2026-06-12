"""Controlled optimizer comparison for scratch MLP training."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import make_xor_classification_data
from src.data.preprocessing import standardize_features, train_val_split
from src.models.mlp import BinaryMLPScratch
from src.optimization.parameter_optimizers import (
    ParameterAdam,
    ParameterMomentum,
    ParameterSGD,
)
from src.train import evaluate_binary_mlp, train_binary_mlp
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_multiple_loss_curves
from src.utils.seed import set_seed


def create_optimizer(
    optimizer_name: str,
    optimizer_config: dict,
):
    """
    Create one fresh generic parameter optimizer from config.
    """
    if optimizer_name == "sgd":
        return ParameterSGD(
            learning_rate=optimizer_config["learning_rate"],
        )
    if optimizer_name == "momentum":
        return ParameterMomentum(
            learning_rate=optimizer_config["learning_rate"],
            beta=optimizer_config["beta"],
        )
    if optimizer_name == "adam":
        return ParameterAdam(
            learning_rate=optimizer_config["learning_rate"],
            beta1=optimizer_config["beta1"],
            beta2=optimizer_config["beta2"],
            epsilon=float(optimizer_config["epsilon"]),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def run_optimizer_experiment(
    optimizer_name: str,
    optimizer_config: dict,
    initial_parameters: dict[str, np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int,
    seed: int,
    num_epochs: int,
    batch_size: int,
    log_every: int,
    threshold: float,
    logger,
) -> dict:
    """
    Train one fresh MLP from the shared initial parameters.
    """
    model = BinaryMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=hidden_dim,
        seed=seed,
    )
    model.set_parameters(initial_parameters)

    optimizer = create_optimizer(
        optimizer_name,
        optimizer_config,
    )

    initial_train_metrics = evaluate_binary_mlp(
        model,
        X_train,
        y_train,
        threshold=threshold,
    )
    initial_val_metrics = evaluate_binary_mlp(
        model,
        X_val,
        y_val,
        threshold=threshold,
    )

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
        log_every=log_every,
        logger=logger,
        threshold=threshold,
    )

    final_train_metrics = evaluate_binary_mlp(
        model,
        X_train,
        y_train,
        threshold=threshold,
    )
    final_val_metrics = evaluate_binary_mlp(
        model,
        X_val,
        y_val,
        threshold=threshold,
    )

    return {
        "optimizer_name": optimizer_name,
        "initial_train_metrics": initial_train_metrics,
        "initial_val_metrics": initial_val_metrics,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "history": history,
        "final_parameters": model.get_parameters(),
    }


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "mlp_optimizer_comparison.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    optimizer_config = config["optimizers"]
    output_config = config["output"]
    log_file = config["logging"]["log_file"]

    train_loss_curve_path = output_config["train_loss_curve_path"]
    val_loss_curve_path = output_config["val_loss_curve_path"]
    val_accuracy_curve_path = output_config["val_accuracy_curve_path"]

    set_seed(seed)
    logger = get_logger("mlp_optimizer_comparison", log_file=log_file)

    logger.info("Experiment: mlp_optimizer_comparison")
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
    logger.info("Controlled comparison boundary:")
    logger.info("Same data: yes")
    logger.info("Same split: yes")
    logger.info("Same standardized inputs: yes")
    logger.info("Same initial parameters: yes")
    logger.info("Same mini-batch order: yes")
    logger.info("Same epoch count: %s", training_config["num_epochs"])
    logger.info("Same update count: validated after training")
    logger.info(
        "Optimizer-specific learning rates: sgd=%s, momentum=%s, adam=%s",
        optimizer_config["sgd"]["learning_rate"],
        optimizer_config["momentum"]["learning_rate"],
        optimizer_config["adam"]["learning_rate"],
    )

    base_model = BinaryMLPScratch(
        n_features=X_train_scaled.shape[1],
        hidden_dim=model_config["hidden_dim"],
        seed=seed,
    )
    initial_parameters = base_model.get_parameters()

    optimizer_names = [
        "sgd",
        "momentum",
        "adam",
    ]
    results = {}

    num_epochs = training_config["num_epochs"]
    batch_size = training_config["batch_size"]
    log_every = training_config["log_every"]
    threshold = training_config["threshold"]

    for optimizer_name in optimizer_names:
        result = run_optimizer_experiment(
            optimizer_name=optimizer_name,
            optimizer_config=optimizer_config[optimizer_name],
            initial_parameters=initial_parameters,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            hidden_dim=model_config["hidden_dim"],
            seed=seed,
            num_epochs=num_epochs,
            batch_size=batch_size,
            log_every=log_every,
            threshold=threshold,
            logger=logger,
        )
        results[optimizer_name] = result

        logger.info("Optimizer: %s", optimizer_name)
        logger.info(
            "Initial train BCE: %.6f",
            result["initial_train_metrics"]["bce"],
        )
        logger.info(
            "Initial validation BCE: %.6f",
            result["initial_val_metrics"]["bce"],
        )
        logger.info("Final train BCE: %.6f", result["final_train_metrics"]["bce"])
        logger.info("Final validation BCE: %.6f", result["final_val_metrics"]["bce"])
        logger.info(
            "Final train accuracy: %.6f",
            result["final_train_metrics"]["accuracy"],
        )
        logger.info(
            "Final validation accuracy: %.6f",
            result["final_val_metrics"]["accuracy"],
        )
        logger.info("Parameter-update count: %s", result["history"]["update_count"])

    update_counts = {
        name: results[name]["history"]["update_count"]
        for name in optimizer_names
    }
    if len(set(update_counts.values())) != 1:
        raise ValueError(f"Optimizer update counts differ: {update_counts}")
    logger.info("All optimizer update counts are equal: %s", update_counts)

    train_bce_histories = {
        name: results[name]["history"]["train_bce"]
        for name in optimizer_names
    }
    val_bce_histories = {
        name: results[name]["history"]["val_bce"]
        for name in optimizer_names
    }
    val_accuracy_histories = {
        name: results[name]["history"]["val_accuracy"]
        for name in optimizer_names
    }

    plot_multiple_loss_curves(
        train_bce_histories,
        output_path=train_loss_curve_path,
        title="MLP Optimizer Comparison: Training BCE",
        xlabel="Epoch",
        ylabel="Binary Cross Entropy Loss",
    )
    plot_multiple_loss_curves(
        val_bce_histories,
        output_path=val_loss_curve_path,
        title="MLP Optimizer Comparison: Validation BCE",
        xlabel="Epoch",
        ylabel="Binary Cross Entropy Loss",
    )
    plot_multiple_loss_curves(
        val_accuracy_histories,
        output_path=val_accuracy_curve_path,
        title="MLP Optimizer Comparison: Validation Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
    )
    logger.info("Saved training BCE curve to: %s", train_loss_curve_path)
    logger.info("Saved validation BCE curve to: %s", val_loss_curve_path)
    logger.info("Saved validation accuracy curve to: %s", val_accuracy_curve_path)
    logger.info("This is a controlled teaching comparison under one configuration.")
    logger.info("It does not establish a universal optimizer ranking.")
    logger.info(
        "Different learning-rate tuning budgets, architectures, datasets, "
        "and compute budgets may change the outcome."
    )

    # Keep preprocessing statistics visible for beginner-friendly inspection.
    _ = mean, std


if __name__ == "__main__":
    main()
