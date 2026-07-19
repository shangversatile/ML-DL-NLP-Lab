"""Analyze confidence thresholding and abstention diagnostics for digits."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.digits import load_digits_dataset
from src.data.preprocessing import stratified_train_val_test_split
from src.evaluation.experiment_registry import (
    append_experiment_record,
    make_experiment_record,
)
from src.evaluation.selective_prediction import (
    choose_threshold_for_target_selective_accuracy,
    threshold_sweep,
    top_k_fallback_metrics,
    validate_confidence_threshold,
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
from src.utils.plotting import (
    plot_abstention_error_tradeoff,
    plot_selective_accuracy_coverage_curve,
)


DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/digits_mlp_augmented.npz")
DEFAULT_OUTPUT_DIR = Path("results/figures")
DEFAULT_REGISTRY_PATH = Path("results/registry/week5_abstention.jsonl")
DEFAULT_CANVAS_SAMPLES_DIR = Path("data/user_digits/samples")
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

ABSTENTION_CONDITIONS = [
    {"name": "clean", "type": "identity"},
    {"name": "shift_down_1", "type": "shift", "row_shift": 1, "col_shift": 0},
    {"name": "shift_right_1", "type": "shift", "row_shift": 0, "col_shift": 1},
    {"name": "thicken", "type": "thicken"},
    {"name": "noise_0_15", "type": "noise", "noise_std": 0.15},
]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_thresholds(thresholds_text: str | None) -> list[float] | None:
    if thresholds_text is None:
        return None

    values = []
    for item in thresholds_text.split(","):
        stripped = item.strip()
        if not stripped:
            raise ValueError("--thresholds must not contain empty entries.")
        values.append(validate_confidence_threshold(float(stripped)))

    if len(values) == 0:
        raise ValueError("--thresholds must contain at least one threshold.")

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate confidence thresholding and abstention diagnostics.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Checkpoint to evaluate (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Optional comma-separated confidence thresholds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for abstention figures (default: {DEFAULT_OUTPUT_DIR})",
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
        help=(
            "Directory containing labeled canvas samples "
            f"(default: {DEFAULT_CANVAS_SAMPLES_DIR})"
        ),
    )
    parser.add_argument(
        "--target-selective-accuracy",
        type=float,
        default=0.95,
        help="Target selective accuracy for threshold selection.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.50,
        help="Minimum coverage required for threshold selection.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k fallback size for abstained examples.",
    )
    return parser.parse_args()


def _format_metric(value: float | int) -> str:
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.6f}"
    return str(value)


def _registry_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    registry_metrics: dict[str, float | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            registry_metrics[key] = None
        else:
            registry_metrics[key] = value
    return registry_metrics


def _fallback_for_threshold(
    fallback_rows: list[dict[str, float | int]],
    threshold: float,
) -> dict[str, float | int]:
    for row in fallback_rows:
        if float(row["threshold"]) == float(threshold):
            return row
    raise ValueError("fallback row missing for selected threshold.")


def _evaluate_split(
    *,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    condition_name: str,
    dataset: str,
    split: str,
    checkpoint_path: Path,
    output_dir: Path,
    registry_path: Path,
    thresholds: list[float] | None,
    target_selective_accuracy: float,
    min_coverage: float,
    top_k: int,
    notes: str,
    tags: list[str],
) -> dict[str, Any]:
    sweep_results = threshold_sweep(probabilities, y_true, thresholds=thresholds)
    fallback_rows = [
        top_k_fallback_metrics(
            probabilities,
            y_true,
            threshold=float(row["threshold"]),
            k=top_k,
        )
        for row in sweep_results
    ]

    selected = choose_threshold_for_target_selective_accuracy(
        sweep_results,
        target_accuracy=target_selective_accuracy,
        min_coverage=min_coverage,
    )

    selective_curve_path = (
        output_dir / f"digits_abstention_{condition_name}_selective_curve.png"
    )
    tradeoff_path = output_dir / f"digits_abstention_{condition_name}_tradeoff.png"
    plot_selective_accuracy_coverage_curve(
        sweep_results,
        str(selective_curve_path),
        title=f"Digits Abstention: {condition_name}",
    )
    plot_abstention_error_tradeoff(
        sweep_results,
        str(tradeoff_path),
        title=f"Digits Abstention Tradeoff: {condition_name}",
    )

    if selected is not None:
        selected_fallback = _fallback_for_threshold(
            fallback_rows,
            float(selected["threshold"]),
        )
        record_metrics = {
            **selected,
            "target_selective_accuracy": target_selective_accuracy,
            "min_coverage": min_coverage,
            "top_k": top_k,
            "abstained_top_k_hits": selected_fallback[
                "abstained_top_k_hits"
            ],
            "abstained_top_k_hit_rate": selected_fallback[
                "abstained_top_k_hit_rate"
            ],
            "answered_top1_accuracy": selected_fallback[
                "answered_top1_accuracy"
            ],
        }
        record = make_experiment_record(
            name=f"digits_abstention_{condition_name}",
            split=split,
            metrics=_registry_metrics(record_metrics),
            model="MulticlassMLPScratch",
            dataset=dataset,
            checkpoint=str(checkpoint_path),
            notes=notes,
            tags=tags,
        )
        append_experiment_record(record, registry_path)

    return {
        "dataset_split": f"{dataset}/{split}",
        "condition_name": condition_name,
        "sweep_results": sweep_results,
        "fallback_rows": fallback_rows,
        "selected": selected,
        "selective_curve_path": selective_curve_path,
        "tradeoff_path": tradeoff_path,
    }


def _print_sweep_table(results: list[dict[str, Any]]) -> None:
    headers = (
        "dataset/split",
        "threshold",
        "coverage",
        "selective_acc",
        "err_answer",
        "err_abstain",
        "err_abst_rate",
        "abst_topk_hit",
    )
    print(
        (
            f"{headers[0]:<42} {headers[1]:>9} {headers[2]:>9} "
            f"{headers[3]:>13} {headers[4]:>10} {headers[5]:>11} "
            f"{headers[6]:>13} {headers[7]:>13}"
        )
    )
    print("-" * 128)
    for result in results:
        dataset_split = str(result["dataset_split"])
        for sweep_row, fallback_row in zip(
            result["sweep_results"],
            result["fallback_rows"],
        ):
            print(
                (
                    f"{dataset_split:<42} "
                    f"{_format_metric(float(sweep_row['threshold'])):>9} "
                    f"{_format_metric(float(sweep_row['coverage'])):>9} "
                    f"{_format_metric(float(sweep_row['selective_accuracy'])):>13} "
                    f"{int(sweep_row['errors_answered']):>10} "
                    f"{int(sweep_row['errors_abstained']):>11} "
                    f"{_format_metric(float(sweep_row['error_abstention_rate'])):>13} "
                    f"{_format_metric(float(fallback_row['abstained_top_k_hit_rate'])):>13}"
                )
            )


def _print_selected_thresholds(
    results: list[dict[str, Any]],
    target_selective_accuracy: float,
    min_coverage: float,
) -> None:
    print("")
    print(
        "Selected thresholds "
        f"(target_selective_accuracy={target_selective_accuracy:.6f}, "
        f"min_coverage={min_coverage:.6f}):"
    )
    for result in results:
        selected = result["selected"]
        if selected is None:
            print(
                f"{result['dataset_split']}: no threshold satisfies target "
                "and minimum coverage."
            )
            continue

        print(
            (
                f"{result['dataset_split']}: threshold="
                f"{float(selected['threshold']):.6f}, coverage="
                f"{float(selected['coverage']):.6f}, selective_accuracy="
                f"{float(selected['selective_accuracy']):.6f}, "
                f"errors_answered={int(selected['errors_answered'])}, "
                f"errors_abstained={int(selected['errors_abstained'])}"
            )
        )


def _print_artifacts(results: list[dict[str, Any]], registry_path: Path) -> None:
    print("")
    print("Artifacts:")
    for result in results:
        print(f"{result['condition_name']}_selective_curve: {result['selective_curve_path']}")
        print(f"{result['condition_name']}_tradeoff: {result['tradeoff_path']}")
    print(f"registry: {registry_path}")


def main() -> None:
    args = parse_args()
    thresholds = _parse_thresholds(args.thresholds)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

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

    results: list[dict[str, Any]] = []
    for index, condition in enumerate(ABSTENTION_CONDITIONS):
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
        results.append(
            _evaluate_split(
                probabilities=probabilities,
                y_true=y_test,
                condition_name=condition_name,
                dataset=dataset,
                split=split,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                registry_path=registry_path,
                thresholds=thresholds,
                target_selective_accuracy=args.target_selective_accuracy,
                min_coverage=args.min_coverage,
                top_k=args.top_k,
                notes=(
                    "Confidence thresholding and abstention diagnostics only; "
                    "no retraining, fine-tuning, or model math changes applied."
                ),
                tags=["week5", "task7c", "abstention", condition_name],
            )
        )

    if args.include_canvas:
        samples_dir = _project_path(args.canvas_samples_dir)
        samples = load_canvas_samples(samples_dir, require_labels=True)
        if len(samples) == 0:
            print(f"No labeled canvas samples found under {samples_dir}; skipping canvas.")
        else:
            print(
                "Canvas-Diagnostic-v1 is diagnostic-only; do not use selected "
                "thresholds from this split as final policy choices."
            )
            X_canvas, y_canvas = stack_canvas_sample_features(samples)
            canvas_probabilities = model.predict_proba(X_canvas)
            results.append(
                _evaluate_split(
                    probabilities=canvas_probabilities,
                    y_true=y_canvas,
                    condition_name="canvas",
                    dataset="Canvas-Diagnostic-v1",
                    split="diagnostic_canvas",
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    registry_path=registry_path,
                    thresholds=thresholds,
                    target_selective_accuracy=args.target_selective_accuracy,
                    min_coverage=args.min_coverage,
                    top_k=args.top_k,
                    notes=(
                        "Canvas-Diagnostic-v1 abstention diagnostic only; "
                        "not used for training, checkpoint selection, or final "
                        "policy threshold selection."
                    ),
                    tags=[
                        "week5",
                        "task7c",
                        "abstention",
                        "canvas",
                        "diagnostic",
                    ],
                )
            )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint model class: {metadata['model_class']}")
    print(f"Figures: {output_dir}")
    print(f"Registry: {registry_path}")
    print("")
    _print_sweep_table(results)
    _print_selected_thresholds(
        results,
        args.target_selective_accuracy,
        args.min_coverage,
    )
    _print_artifacts(results, registry_path)


if __name__ == "__main__":
    main()
