"""Evaluate saved user-drawn canvas digit samples with a checkpointed MLP."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.canvas_sample_store import (
    load_canvas_samples,
    stack_canvas_sample_features,
)
from src.models.checkpoint import load_multiclass_mlp_checkpoint
from src.utils.multiclass import multiclass_cross_entropy
from src.evaluation.canvas_diagnostics import (
    canvas_confusion_matrix,
    high_confidence_canvas_errors,
    per_class_canvas_summary,
    summarize_canvas_validation,
    top_k_miss_errors,
)
from src.evaluation.multiclass_metrics import prediction_confidence
from src.utils.plotting import plot_canvas_confusion_matrix, plot_canvas_error_grid


DEFAULT_SAMPLES_DIR = Path("data/user_digits/samples")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/digits_mlp_augmented.npz")
DEFAULT_OUTPUT_DIR = Path("results/canvas_debug")


def _project_path(
    path: Path,
) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate labeled user-drawn canvas digit samples.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help=f"Directory containing saved canvas samples (default: {DEFAULT_SAMPLES_DIR})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Checkpoint to evaluate (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k accuracy value to report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for diagnostic figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.90,
        help="Confidence threshold for high-confidence error reporting.",
    )
    return parser.parse_args()


def _format_metric(
    value: float | int,
) -> str:
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.6f}"
    return str(value)


def _print_overall_summary(
    summary: dict[str, float | int],
    cross_entropy: float,
    top_k: int,
) -> None:
    print("Overall summary:")
    print(f"n_samples: {summary['n_samples']}")
    print(f"n_errors: {summary['n_errors']}")
    print(f"accuracy: {summary['accuracy']:.6f}")
    print(f"top_{top_k}_accuracy: {summary['top_k_accuracy']:.6f}")
    print(f"cross_entropy: {cross_entropy:.6f}")
    print(f"mean_confidence: {summary['mean_confidence']:.6f}")
    print(
        f"mean_confidence_correct: "
        f"{_format_metric(summary['mean_confidence_correct'])}"
    )
    print(
        f"mean_confidence_incorrect: "
        f"{_format_metric(summary['mean_confidence_incorrect'])}"
    )
    print(f"overconfidence_gap: {summary['overconfidence_gap']:.6f}")
    print(f"top_k_miss_error_count: {summary['top_k_miss_error_count']}")
    print(f"top_k_miss_error_rate: {summary['top_k_miss_error_rate']:.6f}")


def _print_per_class_summary(
    summary: list[dict[str, float | int]],
    top_k: int,
) -> None:
    print("Per-class summary:")
    print(
        (
            "class | count | correct | errors | accuracy | "
            f"top_{top_k}_accuracy | mean_confidence | mean_error_confidence"
        )
    )
    for record in summary:
        print(
            (
                f"{record['class_label']:>5} | "
                f"{record['count']:>5} | "
                f"{record['correct']:>7} | "
                f"{record['errors']:>6} | "
                f"{_format_metric(record['accuracy']):>8} | "
                f"{_format_metric(record['top_k_accuracy']):>14} | "
                f"{_format_metric(record['mean_confidence']):>15} | "
                f"{_format_metric(record['mean_error_confidence']):>21}"
            )
        )


def _print_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
) -> None:
    print("Confusion matrix (rows=true, columns=predicted):")
    print("true\\pred | " + " ".join(f"{name:>4}" for name in class_names))
    print("-" * (11 + 5 * len(class_names)))
    for row_index, row in enumerate(matrix):
        print(
            f"{class_names[row_index]:>9} | "
            + " ".join(f"{int(value):>4}" for value in row)
        )


def _print_error_records(
    title: str,
    records: list[dict],
) -> None:
    print(title)
    if len(records) == 0:
        print("none")
        return

    for record in records:
        print(
            (
                f"{Path(record['path']).name} | "
                f"true={int(record['true_label'])} | "
                f"pred={int(record['prediction'])} | "
                f"confidence={float(record['confidence']):.6f} | "
                f"top_k={np.asarray(record['top_k_indices'], dtype=int).tolist()}"
            )
        )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    samples_dir = _project_path(args.samples_dir)
    samples = load_canvas_samples(samples_dir, require_labels=True)
    if len(samples) == 0:
        print("No labeled canvas samples found. Use the app to save labeled samples first.")
        return

    checkpoint_path = _project_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Run python experiments/compare_digits_augmented_training.py "
            "to create the augmented checkpoint."
        )

    model, metadata = load_multiclass_mlp_checkpoint(checkpoint_path)
    X, y = stack_canvas_sample_features(samples)
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)
    confidences = prediction_confidence(probabilities)
    top_k = min(args.top_k, probabilities.shape[1])
    top_k_indices = np.argsort(-probabilities, axis=1)[:, :top_k]

    cross_entropy = float(multiclass_cross_entropy(y, probabilities))
    overall_summary = summarize_canvas_validation(
        y,
        predictions,
        probabilities,
        top_k_indices,
    )
    per_class_summary = per_class_canvas_summary(
        y,
        predictions,
        top_k_indices,
        confidences,
        num_classes=probabilities.shape[1],
    )
    confusion_matrix = canvas_confusion_matrix(
        y,
        predictions,
        num_classes=probabilities.shape[1],
    )

    class_names = [
        str(class_name)
        for class_name in metadata.get(
            "class_names",
            [str(index) for index in range(probabilities.shape[1])],
        )
    ]
    output_dir = _project_path(args.output_dir)
    confusion_matrix_path = output_dir / "canvas_confusion_matrix.png"
    high_confidence_errors_path = output_dir / "canvas_high_confidence_errors.png"
    top_k_miss_errors_path = output_dir / "canvas_top_k_miss_errors.png"

    sample_records = []
    for sample, true_label, prediction, confidence, row_top_k_indices in zip(
        samples,
        y,
        predictions,
        confidences,
        top_k_indices,
    ):
        sample_records.append(
            {
                "path": sample["path"],
                "resized_8x8": sample["resized_8x8"],
                "true_label": int(true_label),
                "prediction": int(prediction),
                "confidence": float(confidence),
                "top_k_indices": row_top_k_indices.copy(),
            }
        )

    high_confidence_errors = high_confidence_canvas_errors(
        sample_records,
        threshold=args.high_confidence_threshold,
    )
    top_k_misses = top_k_miss_errors(sample_records)

    plot_canvas_confusion_matrix(
        confusion_matrix,
        str(confusion_matrix_path),
        class_names=class_names,
    )
    plot_canvas_error_grid(
        high_confidence_errors,
        str(high_confidence_errors_path),
        title=(
            "Real Canvas High-Confidence Errors "
            f"(confidence >= {args.high_confidence_threshold:.2f})"
        ),
    )
    plot_canvas_error_grid(
        top_k_misses,
        str(top_k_miss_errors_path),
        title=f"Real Canvas Top-{top_k} Miss Errors",
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples directory: {samples_dir}")
    print("")
    print("Per-sample results:")
    for sample, true_label, prediction, confidence, row_probabilities, row_top_k in zip(
        samples,
        y,
        predictions,
        confidences,
        probabilities,
        top_k_indices,
    ):
        top_k_probabilities = row_probabilities[row_top_k]
        correctness = "correct" if prediction == true_label else "incorrect"
        saved_prediction = sample.get("saved_prediction")
        saved_match = (
            "same-as-saved"
            if saved_prediction is not None and int(saved_prediction) == int(prediction)
            else "changed-from-saved"
        )
        print(
            (
                f"{sample['path'].name} | true={class_names[int(true_label)]} | "
                f"pred={class_names[int(prediction)]} | confidence={float(confidence):.6f} | "
                f"top_{top_k}={row_top_k.tolist()} | "
                f"top_{top_k}_probs={[float(value) for value in top_k_probabilities]} | "
                f"{correctness} | {saved_match}"
            )
        )

    print("")
    _print_overall_summary(overall_summary, cross_entropy, top_k)
    print("")
    _print_per_class_summary(per_class_summary, top_k)
    print("")
    _print_confusion_matrix(confusion_matrix, class_names)
    print("")
    _print_error_records(
        f"High-confidence errors (confidence >= {args.high_confidence_threshold:.2f}):",
        high_confidence_errors,
    )
    print("")
    _print_error_records(f"Top-{top_k} miss errors:", top_k_misses)
    print("")
    print("Artifacts:")
    print(f"canvas_confusion_matrix: {confusion_matrix_path}")
    print(f"canvas_high_confidence_errors: {high_confidence_errors_path}")
    print(f"canvas_top_k_miss_errors: {top_k_miss_errors_path}")
    if len(samples) < 30:
        print("Warning: sample size is too small for stable conclusions.")


if __name__ == "__main__":
    main()
