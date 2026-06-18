"""Compare clean-only and augmented training under synthetic digit shifts."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.digit_augmentation import create_augmented_digit_dataset
from src.data.digits import load_digits_dataset
from src.data.preprocessing import stratified_train_val_test_split
from src.evaluation.confidence_diagnostics import (
    confidence_bin_summary,
    expected_calibration_error,
)
from src.evaluation.multiclass_metrics import (
    prediction_confidence,
    top_k_accuracy,
)
from src.evaluation.shift_diagnostics import (
    apply_shift_condition,
    flatten_digit_images,
)
from src.models.checkpoint import (
    build_multiclass_mlp_metadata,
    save_multiclass_mlp_checkpoint,
)
from src.models.multiclass_mlp import MulticlassMLPScratch
from src.optimization.parameter_optimizers import ParameterAdam
from src.train import (
    evaluate_multiclass_mlp,
    train_multiclass_mlp_fixed_updates,
)
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.multiclass import multiclass_cross_entropy
from src.utils.plotting import plot_grouped_metric_bars
from src.utils.seed import set_seed


def create_adam(
    optimizer_config: dict,
) -> ParameterAdam:
    if optimizer_config["name"].lower() != "adam":
        raise ValueError("Only the Adam optimizer is configured for this experiment.")

    return ParameterAdam(
        learning_rate=optimizer_config["learning_rate"],
        beta1=optimizer_config["beta1"],
        beta2=optimizer_config["beta2"],
        epsilon=float(optimizer_config["epsilon"]),
    )


def evaluate_model_under_shift_conditions(
    model,
    X_test_images: np.ndarray,
    y_test: np.ndarray,
    shift_conditions: list[dict],
    seed: int,
    n_bins: int,
    top_k: int,
) -> list[dict[str, float | str]]:
    results = []

    for condition_index, condition in enumerate(shift_conditions):
        condition_name = condition["name"]
        shifted_images = apply_shift_condition(
            X_test_images,
            condition,
            seed=seed + condition_index,
        )
        X_shifted = flatten_digit_images(shifted_images)
        probabilities = model.predict_proba(X_shifted)
        predictions = np.argmax(probabilities, axis=1)
        confidences = prediction_confidence(probabilities)
        bin_summary = confidence_bin_summary(
            y_test,
            probabilities,
            n_bins=n_bins,
        )

        results.append(
            {
                "condition": condition_name,
                "accuracy": float(np.mean(predictions == y_test)),
                "cross_entropy": float(
                    multiclass_cross_entropy(y_test, probabilities)
                ),
                "mean_confidence": float(np.mean(confidences)),
                "ece": float(expected_calibration_error(bin_summary, len(y_test))),
                "top_k_accuracy": float(
                    top_k_accuracy(
                        y_test,
                        probabilities,
                        k=min(top_k, probabilities.shape[1]),
                    )
                ),
            }
        )

    return results


def _result_by_condition(
    results: list[dict[str, float | str]],
    condition_name: str,
) -> dict[str, float | str]:
    for result in results:
        if result["condition"] == condition_name:
            return result
    raise ValueError(f"missing condition result: {condition_name}")


def _log_condition_table(
    logger,
    baseline_results: list[dict[str, float | str]],
    augmented_results: list[dict[str, float | str]],
) -> None:
    logger.info(
        (
            "condition | baseline_acc | augmented_acc | delta_acc | "
            "baseline_ce | augmented_ce | delta_ce | baseline_ece | "
            "augmented_ece | delta_ece"
        )
    )
    for baseline_result, augmented_result in zip(
        baseline_results,
        augmented_results,
    ):
        delta_accuracy = (
            float(augmented_result["accuracy"])
            - float(baseline_result["accuracy"])
        )
        delta_cross_entropy = (
            float(augmented_result["cross_entropy"])
            - float(baseline_result["cross_entropy"])
        )
        delta_ece = float(augmented_result["ece"]) - float(baseline_result["ece"])
        logger.info(
            (
                "%s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | "
                "%.6f | %.6f | %.6f"
            ),
            baseline_result["condition"],
            baseline_result["accuracy"],
            augmented_result["accuracy"],
            delta_accuracy,
            baseline_result["cross_entropy"],
            augmented_result["cross_entropy"],
            delta_cross_entropy,
            baseline_result["ece"],
            augmented_result["ece"],
            delta_ece,
        )


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "digits_augmented_robustness.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    optimizer_config = config["optimizer"]
    augmentation_conditions = config["augmentation_conditions"]
    evaluation_shift_conditions = config["evaluation_shift_conditions"]
    analysis_config = config["analysis"]
    checkpoint_config = config["checkpoint"]
    output_config = config["output"]

    set_seed(seed)
    logger = get_logger(
        "digits_augmented_robustness",
        log_file=config["logging"]["log_file"],
    )
    logger.info("Experiment: digits_augmented_robustness")
    logger.info("Seed: %s", seed)

    X, y, _ = load_digits_dataset(
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
    X_train_images = X_train.reshape(-1, 8, 8)
    X_test_images = X_test.reshape(-1, 8, 8)

    augmented_dataset = create_augmented_digit_dataset(
        X_train_images,
        y_train,
        augmentation_conditions,
        seed=seed,
    )
    logger.info("Original train size: %s", X_train.shape[0])
    logger.info("Augmented train size: %s", augmented_dataset["X"].shape[0])
    logger.info(
        "Augmentation multiplier: %.2f",
        augmented_dataset["X"].shape[0] / X_train.shape[0],
    )
    logger.info(
        "Augmentation conditions: %s",
        [condition["name"] for condition in augmentation_conditions],
    )

    initial_model = MulticlassMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        seed=seed,
    )
    initial_parameters = initial_model.get_parameters()

    baseline_model = MulticlassMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        seed=seed + 1,
    )
    baseline_model.set_parameters(initial_parameters)
    baseline_optimizer = create_adam(optimizer_config)
    logger.info("Training clean-only baseline for fixed update budget.")
    train_multiclass_mlp_fixed_updates(
        baseline_model,
        baseline_optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        max_updates=training_config["max_updates"],
        batch_size=training_config["batch_size"],
        seed=seed,
        eval_every_updates=training_config["eval_every_updates"],
        log_every_updates=training_config["log_every_updates"],
        logger=logger,
    )

    augmented_model = MulticlassMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        seed=seed + 2,
    )
    augmented_model.set_parameters(initial_parameters)
    augmented_optimizer = create_adam(optimizer_config)
    logger.info("Training augmented model for the same fixed update budget.")
    train_multiclass_mlp_fixed_updates(
        augmented_model,
        augmented_optimizer,
        augmented_dataset["X"],
        augmented_dataset["y"],
        X_val,
        y_val,
        max_updates=training_config["max_updates"],
        batch_size=training_config["batch_size"],
        seed=seed,
        eval_every_updates=training_config["eval_every_updates"],
        log_every_updates=training_config["log_every_updates"],
        logger=logger,
    )

    baseline_results = evaluate_model_under_shift_conditions(
        baseline_model,
        X_test_images,
        y_test,
        evaluation_shift_conditions,
        seed=seed,
        n_bins=analysis_config["n_bins"],
        top_k=analysis_config["top_k"],
    )
    augmented_results = evaluate_model_under_shift_conditions(
        augmented_model,
        X_test_images,
        y_test,
        evaluation_shift_conditions,
        seed=seed,
        n_bins=analysis_config["n_bins"],
        top_k=analysis_config["top_k"],
    )

    baseline_clean = _result_by_condition(baseline_results, "clean")
    augmented_clean = _result_by_condition(augmented_results, "clean")
    baseline_thicken = _result_by_condition(baseline_results, "thicken")
    augmented_thicken = _result_by_condition(augmented_results, "thicken")
    baseline_worst = min(
        baseline_results,
        key=lambda result: float(result["accuracy"]),
    )
    augmented_worst = min(
        augmented_results,
        key=lambda result: float(result["accuracy"]),
    )

    logger.info(
        (
            "Clean tradeoff - baseline_acc: %.6f augmented_acc: %.6f "
            "delta_acc: %.6f baseline_ce: %.6f augmented_ce: %.6f "
            "delta_ce: %.6f"
        ),
        baseline_clean["accuracy"],
        augmented_clean["accuracy"],
        float(augmented_clean["accuracy"]) - float(baseline_clean["accuracy"]),
        baseline_clean["cross_entropy"],
        augmented_clean["cross_entropy"],
        float(augmented_clean["cross_entropy"])
        - float(baseline_clean["cross_entropy"]),
    )
    logger.info(
        (
            "Thicken comparison - baseline_acc: %.6f augmented_acc: %.6f "
            "delta_acc: %.6f baseline_ce: %.6f augmented_ce: %.6f "
            "delta_ce: %.6f baseline_ece: %.6f augmented_ece: %.6f "
            "delta_ece: %.6f"
        ),
        baseline_thicken["accuracy"],
        augmented_thicken["accuracy"],
        float(augmented_thicken["accuracy"]) - float(baseline_thicken["accuracy"]),
        baseline_thicken["cross_entropy"],
        augmented_thicken["cross_entropy"],
        float(augmented_thicken["cross_entropy"])
        - float(baseline_thicken["cross_entropy"]),
        baseline_thicken["ece"],
        augmented_thicken["ece"],
        float(augmented_thicken["ece"]) - float(baseline_thicken["ece"]),
    )
    logger.info(
        "Baseline worst shift: %s accuracy=%.6f CE=%.6f ECE=%.6f",
        baseline_worst["condition"],
        baseline_worst["accuracy"],
        baseline_worst["cross_entropy"],
        baseline_worst["ece"],
    )
    logger.info(
        "Augmented worst shift: %s accuracy=%.6f CE=%.6f ECE=%.6f",
        augmented_worst["condition"],
        augmented_worst["accuracy"],
        augmented_worst["cross_entropy"],
        augmented_worst["ece"],
    )
    _log_condition_table(logger, baseline_results, augmented_results)

    final_train_metrics = evaluate_multiclass_mlp(
        augmented_model,
        augmented_dataset["X"],
        augmented_dataset["y"],
    )
    final_validation_metrics = evaluate_multiclass_mlp(
        augmented_model,
        X_val,
        y_val,
    )
    metadata = build_multiclass_mlp_metadata(
        model=augmented_model,
        input_scaling=checkpoint_config["input_scaling"],
        class_names=checkpoint_config["class_names"],
        extra_metadata={
            "seed": seed,
            "max_updates": training_config["max_updates"],
            "batch_size": training_config["batch_size"],
            "optimizer": optimizer_config,
            "augmentation_condition_names": [
                condition["name"] for condition in augmentation_conditions
            ],
            "final_augmented_train_metrics": final_train_metrics,
            "final_validation_metrics": final_validation_metrics,
            "clean_metrics": augmented_clean,
            "thicken_metrics": augmented_thicken,
            "worst_shift_metrics": augmented_worst,
        },
    )
    save_multiclass_mlp_checkpoint(
        augmented_model,
        PROJECT_ROOT / checkpoint_config["augmented_path"],
        metadata,
    )
    logger.info("Saved augmented checkpoint to: %s", checkpoint_config["augmented_path"])

    condition_names = [result["condition"] for result in baseline_results]
    plot_grouped_metric_bars(
        condition_names,
        {
            "Baseline": [float(result["accuracy"]) for result in baseline_results],
            "Augmented": [float(result["accuracy"]) for result in augmented_results],
        },
        output_path=output_config["accuracy_comparison_path"],
        ylabel="Accuracy",
        title="Baseline vs Augmented Accuracy Under Shift",
    )
    plot_grouped_metric_bars(
        condition_names,
        {
            "Baseline": [
                float(result["cross_entropy"]) for result in baseline_results
            ],
            "Augmented": [
                float(result["cross_entropy"]) for result in augmented_results
            ],
        },
        output_path=output_config["cross_entropy_comparison_path"],
        ylabel="Cross Entropy",
        title="Baseline vs Augmented Cross Entropy Under Shift",
    )
    plot_grouped_metric_bars(
        condition_names,
        {
            "Baseline": [float(result["ece"]) for result in baseline_results],
            "Augmented": [float(result["ece"]) for result in augmented_results],
        },
        output_path=output_config["ece_comparison_path"],
        ylabel="ECE-style diagnostic",
        title="Baseline vs Augmented ECE-style Diagnostic Under Shift",
    )
    logger.info("Saved accuracy comparison to: %s", output_config["accuracy_comparison_path"])
    logger.info(
        "Saved cross entropy comparison to: %s",
        output_config["cross_entropy_comparison_path"],
    )
    logger.info("Saved ECE comparison to: %s", output_config["ece_comparison_path"])
    logger.info(
        (
            "This experiment tests whether app-like augmentation improves robustness. "
            "It is not a proof of general robustness and may overfit the chosen "
            "synthetic probes. MNIST and convolutional inductive bias remain future "
            "extensions."
        )
    )


if __name__ == "__main__":
    main()
