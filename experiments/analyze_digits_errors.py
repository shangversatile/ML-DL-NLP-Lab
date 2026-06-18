"""Analyze multiclass MLP errors on the handwritten digits baseline."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.digits import load_digits_dataset
from src.data.preprocessing import stratified_train_val_test_split
from src.evaluation.error_analysis import (
    select_high_confidence_errors,
    select_top_loss_examples,
    summarize_errors,
)
from src.evaluation.multiclass_metrics import (
    accuracy_from_confusion_matrix,
    confusion_matrix,
    macro_average,
    per_class_precision,
    per_class_recall,
    top_k_accuracy,
)
from src.models.multiclass_mlp import MulticlassMLPScratch
from src.optimization.parameter_optimizers import ParameterAdam
from src.train import evaluate_multiclass_mlp, train_multiclass_mlp
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.plotting import plot_confusion_matrix, plot_digit_examples
from src.utils.seed import set_seed


def create_adam(optimizer_config: dict) -> ParameterAdam:
    if optimizer_config["name"].lower() != "adam":
        raise ValueError("Only the Adam optimizer is configured for this experiment.")

    return ParameterAdam(
        learning_rate=optimizer_config["learning_rate"],
        beta1=optimizer_config["beta1"],
        beta2=optimizer_config["beta2"],
        epsilon=float(optimizer_config["epsilon"]),
    )


def summarize_class_counts(
    y: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    return np.bincount(y, minlength=num_classes)


def format_error_title(
    example: dict[str, float | int],
) -> str:
    return (
        f"idx={example['index']}, y={example['true_label']}, "
        f"pred={example['predicted_label']}\n"
        f"conf={example['confidence']:.2f}, "
        f"p_true={example['true_class_probability']:.2f}\n"
        f"nll={example['negative_log_likelihood']:.2f}"
    )


def _log_metric_dict(
    logger,
    heading: str,
    metrics: dict[str, float | int],
) -> None:
    logger.info("%s", heading)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info("  %s: %.6f", key, value)
        else:
            logger.info("  %s: %s", key, value)


def _log_per_class_table(
    logger,
    heading: str,
    values: np.ndarray,
) -> None:
    logger.info("%s", heading)
    for class_index, value in enumerate(values):
        logger.info("  class %s: %.6f", class_index, value)


def _log_example_records(
    logger,
    heading: str,
    records: list[dict[str, float | int]],
) -> None:
    logger.info("%s", heading)
    if not records:
        logger.info("  none")
        return

    for record in records:
        logger.info(
            (
                "  index=%s true=%s predicted=%s confidence=%.6f "
                "true_probability=%.6f nll=%.6f is_error=%s"
            ),
            record["index"],
            record["true_label"],
            record["predicted_label"],
            record["confidence"],
            record["true_class_probability"],
            record["negative_log_likelihood"],
            record["is_error"],
        )


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "digits_error_analysis.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    optimizer_config = config["optimizer"]
    analysis_config = config["analysis"]
    output_config = config["output"]
    log_file = config["logging"]["log_file"]

    set_seed(seed)
    logger = get_logger("digits_error_analysis", log_file=log_file)

    logger.info("Experiment: digits_error_analysis")
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
    X_test_images = X_test.reshape(-1, 8, 8)
    num_classes = model_config["num_classes"]

    logger.info("Dataset shape: X=%s, images=%s, y=%s", X.shape, images.shape, y.shape)
    logger.info("Train shape: X=%s, y=%s", X_train.shape, y_train.shape)
    logger.info("Validation shape: X=%s, y=%s", X_val.shape, y_val.shape)
    logger.info("Test shape: X=%s, y=%s", X_test.shape, y_test.shape)
    logger.info("Train class counts: %s", summarize_class_counts(y_train, num_classes))
    logger.info(
        "Validation class counts: %s",
        summarize_class_counts(y_val, num_classes),
    )
    logger.info("Test class counts: %s", summarize_class_counts(y_test, num_classes))

    model = MulticlassMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=model_config["hidden_dim"],
        num_classes=num_classes,
        seed=seed,
    )
    optimizer = create_adam(optimizer_config)

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

    test_probabilities = model.predict_proba(X_test)
    test_predictions = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        num_classes,
    )
    matrix_accuracy = accuracy_from_confusion_matrix(matrix)
    recalls = per_class_recall(matrix)
    precisions = per_class_precision(matrix)
    macro_recall = macro_average(recalls)
    macro_precision = macro_average(precisions)
    top_k_accuracies = {
        f"top_{k}_accuracy": top_k_accuracy(y_test, test_probabilities, k)
        for k in analysis_config["top_k_values"]
    }
    error_summary = summarize_errors(
        y_test,
        test_predictions,
        test_probabilities,
        high_confidence_threshold=analysis_config["high_confidence_threshold"],
    )
    max_examples = analysis_config["max_examples_to_plot"]
    top_loss_examples = select_top_loss_examples(
        y_test,
        test_predictions,
        test_probabilities,
        top_n=max_examples,
    )
    high_confidence_errors = select_high_confidence_errors(
        y_test,
        test_predictions,
        test_probabilities,
        threshold=analysis_config["high_confidence_threshold"],
        top_n=max_examples,
    )

    logger.info("Confusion-matrix-derived accuracy: %.6f", matrix_accuracy)
    logger.info("Macro recall: %.6f", macro_recall)
    logger.info("Macro precision: %.6f", macro_precision)
    _log_metric_dict(logger, "Top-k accuracies:", top_k_accuracies)
    _log_metric_dict(logger, "Error summary:", error_summary)
    _log_per_class_table(logger, "Per-class recall:", recalls)
    _log_per_class_table(logger, "Per-class precision:", precisions)
    _log_example_records(logger, "Top-loss example records:", top_loss_examples)
    _log_example_records(
        logger,
        "High-confidence error records:",
        high_confidence_errors,
    )

    class_names = [str(index) for index in range(num_classes)]
    plot_confusion_matrix(
        matrix,
        output_path=output_config["confusion_matrix_counts_path"],
        class_names=class_names,
        normalize=False,
        title="Digits Confusion Matrix: Counts",
    )
    plot_confusion_matrix(
        matrix,
        output_path=output_config["confusion_matrix_normalized_path"],
        class_names=class_names,
        normalize=True,
        title="Digits Confusion Matrix: Row-Normalized",
    )

    top_loss_indices = [int(example["index"]) for example in top_loss_examples]
    top_loss_titles = [format_error_title(example) for example in top_loss_examples]
    plot_digit_examples(
        X_test_images[top_loss_indices],
        top_loss_titles,
        output_path=output_config["top_loss_examples_path"],
        n_cols=5,
        figure_title="Digits Top-Loss Examples",
    )

    high_confidence_indices = [
        int(example["index"])
        for example in high_confidence_errors
    ]
    high_confidence_titles = [
        format_error_title(example)
        for example in high_confidence_errors
    ]
    plot_digit_examples(
        X_test_images[high_confidence_indices],
        high_confidence_titles,
        output_path=output_config["high_confidence_errors_path"],
        n_cols=5,
        figure_title="Digits High-Confidence Errors",
    )

    logger.info(
        "Saved confusion matrix counts to: %s",
        output_config["confusion_matrix_counts_path"],
    )
    logger.info(
        "Saved normalized confusion matrix to: %s",
        output_config["confusion_matrix_normalized_path"],
    )
    logger.info(
        "Saved top-loss examples to: %s",
        output_config["top_loss_examples_path"],
    )
    logger.info(
        "Saved high-confidence errors to: %s",
        output_config["high_confidence_errors_path"],
    )
    logger.info(
        (
            "This analysis diagnoses the trained baseline model. It does not yet "
            "include checkpointing, calibration curves, or local handwritten-input "
            "distribution-shift analysis."
        )
    )


if __name__ == "__main__":
    main()
