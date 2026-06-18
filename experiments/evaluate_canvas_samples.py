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
from src.evaluation.multiclass_metrics import prediction_confidence, top_k_accuracy


DEFAULT_SAMPLES_DIR = Path("data/user_digits/samples")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/digits_mlp_augmented.npz")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    accuracy = float(np.mean(predictions == y))
    cross_entropy = float(multiclass_cross_entropy(y, probabilities))
    top_k_acc = float(top_k_accuracy(y, probabilities, k=top_k))
    mean_confidence = float(np.mean(confidences))
    n_errors = int(np.sum(predictions != y))

    class_names = metadata.get(
        "class_names",
        [str(index) for index in range(probabilities.shape[1])],
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples directory: {samples_dir}")
    print("")
    print("Per-sample results:")
    for sample, true_label, prediction, confidence, row_probabilities in zip(
        samples,
        y,
        predictions,
        confidences,
        probabilities,
    ):
        top_k_indices = np.argsort(-row_probabilities)[:top_k]
        top_k_probabilities = row_probabilities[top_k_indices]
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
                f"top_{top_k}={top_k_indices.tolist()} | "
                f"top_{top_k}_probs={[float(value) for value in top_k_probabilities]} | "
                f"{correctness} | {saved_match}"
            )
        )

    print("")
    print("Summary:")
    print(f"n_samples: {len(samples)}")
    print(f"n_errors: {n_errors}")
    print(f"accuracy: {accuracy:.6f}")
    print(f"top_{top_k}_accuracy: {top_k_acc:.6f}")
    print(f"cross_entropy: {cross_entropy:.6f}")
    print(f"mean_confidence: {mean_confidence:.6f}")
    if len(samples) < 30:
        print("Warning: sample size is too small for stable conclusions.")


if __name__ == "__main__":
    main()
