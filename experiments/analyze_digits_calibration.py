"""Analyze calibration metrics and reliability diagrams for the digits checkpoint."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.digits import load_digits_dataset
from src.data.preprocessing import stratified_train_val_test_split
from src.evaluation.calibration import (
    calibration_bin_summary,
    calibration_summary,
)
from src.evaluation.experiment_registry import (
    append_experiment_record,
    make_experiment_record,
)
from src.evaluation.shift_diagnostics import (
    apply_shift_condition,
    flatten_digit_images,
)
from src.inference.canvas_sample_store import (
    load_canvas_samples,
    stack_canvas_sample_features,
)
from src.models.checkpoint import load_multiclass_mlp_checkpoint
from src.utils.plotting import plot_reliability_diagram


DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/digits_mlp_augmented.npz")
DEFAULT_OUTPUT_DIR = Path("results/figures")
DEFAULT_REGISTRY_PATH = Path("results/registry/week5_calibration.jsonl")
DEFAULT_CANVAS_SAMPLES_DIR = Path("data/user_digits/samples")
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

CALIBRATION_CONDITIONS = [
    {"name": "clean", "type": "identity"},
    {"name": "shift_down_1", "type": "shift", "row_shift": 1, "col_shift": 0},
    {"name": "shift_right_1", "type": "shift", "row_shift": 0, "col_shift": 1},
    {"name": "thicken", "type": "thicken"},
    {"name": "noise_0_15", "type": "noise", "noise_std": 0.15},
]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate calibration diagnostics for the digits checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Checkpoint to evaluate (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of reliability-diagram confidence bins.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for reliability diagrams (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help=f"JSONL registry output path (default: {DEFAULT_REGISTRY_PATH})",
    )
    parser.add_argument(
        "--include-canvas",
        action="store_true",
        help="Also evaluate local labeled canvas samples if present.",
    )
    parser.add_argument(
        "--canvas-samples-dir",
        type=Path,
        default=DEFAULT_CANVAS_SAMPLES_DIR,
        help=f"Directory containing labeled canvas samples (default: {DEFAULT_CANVAS_SAMPLES_DIR})",
    )
    return parser.parse_args()


def _format_metric(value: float | int) -> str:
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.6f}"
    return str(value)


def _registry_metrics(summary: dict[str, float | int]) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    for key, value in summary.items():
        if isinstance(value, float) and not np.isfinite(value):
            metrics[key] = None
        else:
            metrics[key] = value
    return metrics


def _evaluate_and_record(
    *,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    condition_name: str,
    dataset: str,
    split: str,
    checkpoint_path: Path,
    output_dir: Path,
    registry_path: Path,
    n_bins: int,
    notes: str,
    tags: list[str],
) -> dict[str, float | int | str]:
    summary = calibration_summary(probabilities, y_true, n_bins=n_bins)
    bins = calibration_bin_summary(probabilities, y_true, n_bins=n_bins)

    figure_path = output_dir / f"digits_calibration_{condition_name}.png"
    plot_reliability_diagram(
        bins,
        str(figure_path),
        title=f"Digits Calibration: {condition_name}",
    )

    metrics = _registry_metrics(summary)
    metrics["n_bins"] = n_bins
    record = make_experiment_record(
        name=f"digits_calibration_{condition_name}",
        split=split,
        metrics=metrics,
        model="MulticlassMLPScratch",
        dataset=dataset,
        checkpoint=str(checkpoint_path),
        notes=notes,
        tags=tags,
    )
    append_experiment_record(record, registry_path)

    return {
        "dataset_split": f"{dataset}/{split}",
        "figure": str(figure_path),
        **summary,
    }


def _print_summary_table(rows: list[dict[str, float | int | str]]) -> None:
    headers = (
        "dataset/split",
        "n",
        "accuracy",
        "mean_conf",
        "gap",
        "ece",
        "mce",
        "brier",
        "nll",
    )
    print(
        (
            f"{headers[0]:<42} {headers[1]:>5} {headers[2]:>10} "
            f"{headers[3]:>10} {headers[4]:>10} {headers[5]:>10} "
            f"{headers[6]:>10} {headers[7]:>10} {headers[8]:>10}"
        )
    )
    print("-" * 132)
    for row in rows:
        print(
            (
                f"{str(row['dataset_split']):<42} "
                f"{int(row['n_samples']):>5} "
                f"{_format_metric(float(row['accuracy'])):>10} "
                f"{_format_metric(float(row['mean_confidence'])):>10} "
                f"{_format_metric(float(row['overconfidence_gap'])):>10} "
                f"{_format_metric(float(row['ece'])):>10} "
                f"{_format_metric(float(row['mce'])):>10} "
                f"{_format_metric(float(row['brier_score'])):>10} "
                f"{_format_metric(float(row['nll'])):>10}"
            )
        )


def main() -> None:
    args = parse_args()
    if isinstance(args.n_bins, bool) or args.n_bins <= 0:
        raise ValueError("--n-bins must be a positive integer.")

    checkpoint_path = _project_path(args.checkpoint)
    output_dir = _project_path(args.output_dir)
    registry_path = _project_path(args.registry)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Run python experiments/compare_digits_augmented_training.py "
            "to create the augmented checkpoint."
        )

    model, metadata = load_multiclass_mlp_checkpoint(checkpoint_path)
    X, y, _ = load_digits_dataset(scale_pixels=True)
    _, _, X_test, _, _, y_test = stratified_train_val_test_split(
        X,
        y,
        val_ratio=DEFAULT_VAL_RATIO,
        test_ratio=DEFAULT_TEST_RATIO,
        seed=DEFAULT_SEED,
    )
    X_test_images = X_test.reshape(-1, 8, 8)

    rows: list[dict[str, float | int | str]] = []
    for index, condition in enumerate(CALIBRATION_CONDITIONS):
        condition_name = condition["name"]
        shifted_images = apply_shift_condition(
            X_test_images,
            condition,
            seed=DEFAULT_SEED + index,
        )
        X_condition = flatten_digit_images(shifted_images)
        probabilities = model.predict_proba(X_condition)
        dataset = "load_digits" if condition_name == "clean" else "load_digits_shift"
        split = "test_clean" if condition_name == "clean" else f"test_{condition_name}"
        rows.append(
            _evaluate_and_record(
                probabilities=probabilities,
                y_true=y_test,
                condition_name=condition_name,
                dataset=dataset,
                split=split,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                registry_path=registry_path,
                n_bins=args.n_bins,
                notes=(
                    "Calibration measurement only; no retraining, recalibration, "
                    "or threshold tuning applied."
                ),
                tags=["week5", "task7b", "calibration", condition_name],
            )
        )

    if args.include_canvas:
        samples_dir = _project_path(args.canvas_samples_dir)
        samples = load_canvas_samples(samples_dir, require_labels=True)
        if len(samples) == 0:
            print(f"No labeled canvas samples found under {samples_dir}; skipping canvas.")
        else:
            X_canvas, y_canvas = stack_canvas_sample_features(samples)
            canvas_probabilities = model.predict_proba(X_canvas)
            rows.append(
                _evaluate_and_record(
                    probabilities=canvas_probabilities,
                    y_true=y_canvas,
                    condition_name="canvas",
                    dataset="Canvas-Diagnostic-v1",
                    split="diagnostic_canvas",
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    registry_path=registry_path,
                    n_bins=args.n_bins,
                    notes=(
                        "Canvas-Diagnostic-v1 calibration measurement only; "
                        "not used for training, checkpoint selection, or calibration tuning."
                    ),
                    tags=["week5", "task7b", "calibration", "canvas", "diagnostic"],
                )
            )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint model class: {metadata['model_class']}")
    print(f"Reliability diagrams: {output_dir}")
    print(f"Registry: {registry_path}")
    print("")
    _print_summary_table(rows)


if __name__ == "__main__":
    main()
