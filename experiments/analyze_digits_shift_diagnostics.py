"""Analyze synthetic shift and confidence diagnostics for the digits checkpoint."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.digits import load_digits_dataset
from src.data.preprocessing import stratified_train_val_test_split
from src.evaluation.confidence_diagnostics import (
    confidence_bin_summary,
    summarize_confidence_behavior,
)
from src.evaluation.error_analysis import summarize_errors
from src.evaluation.multiclass_metrics import (
    per_sample_negative_log_likelihood,
    prediction_confidence,
    top_k_accuracy,
)
from src.evaluation.shift_diagnostics import (
    apply_shift_condition,
    flatten_digit_images,
)
from src.models.checkpoint import load_multiclass_mlp_checkpoint
from src.train import evaluate_multiclass_mlp
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.multiclass import multiclass_cross_entropy
from src.utils.plotting import (
    plot_confidence_bin_summary,
    plot_shift_metric_bars,
)
from src.utils.seed import set_seed


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


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "digits_shift_diagnostics.yaml"
    config = load_config(str(config_path))

    seed = config["seed"]
    data_config = config["data"]
    model_config = config["model"]
    checkpoint_config = config["checkpoint"]
    analysis_config = config["analysis"]
    shift_conditions = config["shift_conditions"]
    output_config = config["output"]
    log_file = config["logging"]["log_file"]

    set_seed(seed)
    logger = get_logger("digits_shift_diagnostics", log_file=log_file)

    checkpoint_path = PROJECT_ROOT / checkpoint_config["path"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint is missing. Run: "
            "python experiments/train_save_load_digits_mlp.py"
        )

    model, metadata = load_multiclass_mlp_checkpoint(checkpoint_path)
    logger.info("Experiment: digits_shift_diagnostics")
    logger.info("Seed: %s", seed)
    logger.info("Loaded checkpoint: %s", checkpoint_path)
    logger.info("Checkpoint model class: %s", metadata["model_class"])
    logger.info("Checkpoint class names: %s", metadata["class_names"])

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
    X_test_images = X_test.reshape(-1, 8, 8)

    clean_eval_metrics = evaluate_multiclass_mlp(model, X_test, y_test)
    logger.info(
        "Clean evaluate_multiclass_mlp CE: %.6f, accuracy: %.6f",
        clean_eval_metrics["cross_entropy"],
        clean_eval_metrics["accuracy"],
    )
    logger.info(
        "Data split shapes: train=%s validation=%s test=%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    n_bins = analysis_config["n_bins"]
    num_classes = model_config["num_classes"]
    results = []
    probabilities_by_condition = {}

    for index, condition in enumerate(shift_conditions):
        condition_name = condition["name"]
        shifted_images = apply_shift_condition(
            X_test_images,
            condition,
            seed=seed + index,
        )
        X_shifted = flatten_digit_images(shifted_images)
        probabilities = model.predict_proba(X_shifted)
        predictions = np.argmax(probabilities, axis=1)
        confidences = prediction_confidence(probabilities)
        nll = per_sample_negative_log_likelihood(y_test, probabilities)

        accuracy = float(np.mean(predictions == y_test))
        cross_entropy = multiclass_cross_entropy(y_test, probabilities)
        mean_confidence = float(np.mean(confidences))
        condition_result = {
            "name": condition_name,
            "cross_entropy": float(cross_entropy),
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "top_1_accuracy": top_k_accuracy(y_test, probabilities, k=1),
            "mean_nll": float(np.mean(nll)),
        }
        if num_classes >= 3:
            condition_result["top_3_accuracy"] = top_k_accuracy(
                y_test,
                probabilities,
                k=3,
            )

        confidence_summary = summarize_confidence_behavior(
            y_test,
            probabilities,
            n_bins=n_bins,
        )
        error_summary = summarize_errors(
            y_test,
            predictions,
            probabilities,
            high_confidence_threshold=analysis_config[
                "high_confidence_threshold"
            ],
        )

        condition_result["confidence_summary"] = confidence_summary
        condition_result["error_summary"] = error_summary
        results.append(condition_result)
        probabilities_by_condition[condition_name] = probabilities

        logger.info(
            (
                "Condition %s - accuracy: %.6f - cross_entropy: %.6f - "
                "mean_confidence: %.6f - top_1: %.6f"
            ),
            condition_name,
            condition_result["accuracy"],
            condition_result["cross_entropy"],
            condition_result["mean_confidence"],
            condition_result["top_1_accuracy"],
        )

    clean_result = next(result for result in results if result["name"] == "clean")
    clean_accuracy = clean_result["accuracy"]
    clean_cross_entropy = clean_result["cross_entropy"]
    clean_mean_confidence = clean_result["mean_confidence"]

    for result in results:
        result["delta_accuracy"] = result["accuracy"] - clean_accuracy
        result["delta_cross_entropy"] = (
            result["cross_entropy"] - clean_cross_entropy
        )
        result["delta_mean_confidence"] = (
            result["mean_confidence"] - clean_mean_confidence
        )

    worst_shift = min(results, key=lambda result: result["accuracy"])
    clean_bin_summary = confidence_bin_summary(
        y_test,
        probabilities_by_condition["clean"],
        n_bins=n_bins,
    )
    worst_bin_summary = confidence_bin_summary(
        y_test,
        probabilities_by_condition[worst_shift["name"]],
        n_bins=n_bins,
    )

    condition_names = [result["name"] for result in results]
    plot_shift_metric_bars(
        condition_names,
        [result["accuracy"] for result in results],
        output_path=output_config["accuracy_by_shift_path"],
        ylabel="Accuracy",
        title="Digits Accuracy Under Synthetic Shift",
    )
    plot_shift_metric_bars(
        condition_names,
        [result["cross_entropy"] for result in results],
        output_path=output_config["cross_entropy_by_shift_path"],
        ylabel="Cross Entropy",
        title="Digits Cross Entropy Under Synthetic Shift",
    )
    plot_shift_metric_bars(
        condition_names,
        [result["mean_confidence"] for result in results],
        output_path=output_config["mean_confidence_by_shift_path"],
        ylabel="Mean Confidence",
        title="Digits Mean Confidence Under Synthetic Shift",
    )
    plot_confidence_bin_summary(
        clean_bin_summary,
        output_path=output_config["clean_confidence_bins_path"],
        title="Clean Digits Confidence Bins",
    )
    plot_confidence_bin_summary(
        worst_bin_summary,
        output_path=output_config["worst_shift_confidence_bins_path"],
        title=f"{worst_shift['name']} Confidence Bins",
    )

    _log_metric_dict(logger, "Clean metrics:", clean_result)
    _log_metric_dict(logger, f"Worst-shift metrics ({worst_shift['name']}):", worst_shift)

    logger.info("Deltas relative to clean:")
    for result in results:
        logger.info(
            (
                "  %s: delta_accuracy=%.6f delta_cross_entropy=%.6f "
                "delta_mean_confidence=%.6f"
            ),
            result["name"],
            result["delta_accuracy"],
            result["delta_cross_entropy"],
            result["delta_mean_confidence"],
        )

    _log_metric_dict(
        logger,
        "Clean confidence summary:",
        clean_result["confidence_summary"],
    )
    _log_metric_dict(
        logger,
        f"Worst-shift confidence summary ({worst_shift['name']}):",
        worst_shift["confidence_summary"],
    )
    logger.info(
        "ECE-style diagnostics: clean=%.6f worst_shift_%s=%.6f",
        clean_result["confidence_summary"]["ece"],
        worst_shift["name"],
        worst_shift["confidence_summary"]["ece"],
    )
    logger.info("Saved accuracy shift plot to: %s", output_config["accuracy_by_shift_path"])
    logger.info(
        "Saved cross entropy shift plot to: %s",
        output_config["cross_entropy_by_shift_path"],
    )
    logger.info(
        "Saved mean confidence shift plot to: %s",
        output_config["mean_confidence_by_shift_path"],
    )
    logger.info(
        "Saved clean confidence bins to: %s",
        output_config["clean_confidence_bins_path"],
    )
    logger.info(
        "Saved worst-shift confidence bins to: %s",
        output_config["worst_shift_confidence_bins_path"],
    )
    logger.info(
        (
            "These synthetic shift probes are controlled diagnostics. "
            "They do not fully characterize real canvas-input distribution shift. "
            "No calibration correction or abstention rule is applied in this task."
        )
    )


if __name__ == "__main__":
    main()
