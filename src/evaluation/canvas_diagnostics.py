"""Diagnostics for real user-drawn canvas digit samples."""

from numbers import Integral, Real

import numpy as np


def _validate_num_classes(
    num_classes: int,
) -> None:
    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(
        num_classes,
        Integral,
    ):
        raise TypeError("num_classes must be an integer.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")


def _validate_integer_label_array(
    name: str,
    values: np.ndarray,
    num_classes: int,
    allow_empty: bool = False,
) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not allow_empty and values.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise ValueError(f"{name} must contain integer labels.")
    if values.size > 0 and (np.any(values < 0) or np.any(values >= num_classes)):
        raise ValueError(f"{name} labels must satisfy 0 <= label < num_classes.")


def _validate_confidences(
    confidences: np.ndarray,
    n_samples: int,
) -> None:
    if not isinstance(confidences, np.ndarray):
        raise TypeError("confidences must be a NumPy array.")
    if confidences.ndim != 1:
        raise ValueError("confidences must be one-dimensional.")
    if confidences.shape[0] != n_samples:
        raise ValueError("confidences must match the number of samples.")
    if np.issubdtype(confidences.dtype, np.bool_) or not np.issubdtype(
        confidences.dtype,
        np.number,
    ):
        raise ValueError("confidences must contain numeric values.")
    if not np.all(np.isfinite(confidences)):
        raise ValueError("confidences must contain only finite values.")
    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("confidences must be between 0.0 and 1.0.")


def _validate_top_k_indices(
    top_k_indices: np.ndarray,
    n_samples: int,
    num_classes: int,
) -> None:
    if not isinstance(top_k_indices, np.ndarray):
        raise TypeError("top_k_indices must be a NumPy array.")
    if top_k_indices.ndim != 2:
        raise ValueError("top_k_indices must be two-dimensional.")
    if top_k_indices.shape[0] != n_samples:
        raise ValueError("top_k_indices must match the number of samples.")
    if top_k_indices.shape[1] == 0:
        raise ValueError("top_k_indices must contain at least one candidate.")
    if np.issubdtype(top_k_indices.dtype, np.bool_) or not np.issubdtype(
        top_k_indices.dtype,
        np.integer,
    ):
        raise ValueError("top_k_indices must contain integer labels.")
    if np.any(top_k_indices < 0) or np.any(top_k_indices >= num_classes):
        raise ValueError("top_k_indices labels must satisfy 0 <= label < num_classes.")


def _validate_prediction_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> None:
    _validate_num_classes(num_classes)
    _validate_integer_label_array("y_true", y_true, num_classes)
    _validate_integer_label_array("y_pred", y_pred, num_classes)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have matching sample counts.")


def _validate_probability_matrix(
    probabilities: np.ndarray,
) -> None:
    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities must be a NumPy array.")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional.")
    if probabilities.shape[0] == 0:
        raise ValueError("probabilities must contain at least one sample.")
    if probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one class.")
    if np.issubdtype(probabilities.dtype, np.bool_) or not np.issubdtype(
        probabilities.dtype,
        np.number,
    ):
        raise ValueError("probabilities must contain numeric values.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative.")

    row_sums = np.sum(probabilities, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("probability rows must sum to one.")


def _validate_probability_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    top_k_indices: np.ndarray,
) -> None:
    _validate_probability_matrix(probabilities)
    num_classes = probabilities.shape[1]
    _validate_prediction_inputs(y_true, y_pred, num_classes)
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have matching sample counts.")
    _validate_top_k_indices(top_k_indices, y_true.shape[0], num_classes)


def _validate_confidence_threshold(
    threshold: float,
) -> None:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("threshold must be numeric and not boolean.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")


def _validate_sample_records(
    sample_records: list[dict],
) -> None:
    if not isinstance(sample_records, list):
        raise TypeError("sample_records must be a list.")
    for record in sample_records:
        if not isinstance(record, dict):
            raise TypeError("each sample record must be a dictionary.")
        required_keys = {
            "path",
            "true_label",
            "prediction",
            "confidence",
            "top_k_indices",
        }
        missing_keys = required_keys - set(record)
        if missing_keys:
            missing_text = ", ".join(sorted(missing_keys))
            raise ValueError(f"sample record missing required keys: {missing_text}.")

        true_label = record["true_label"]
        prediction = record["prediction"]
        confidence = record["confidence"]
        if isinstance(true_label, (bool, np.bool_)) or not isinstance(
            true_label,
            Integral,
        ):
            raise TypeError("true_label must be an integer.")
        if isinstance(prediction, (bool, np.bool_)) or not isinstance(
            prediction,
            Integral,
        ):
            raise TypeError("prediction must be an integer.")
        if isinstance(confidence, (bool, np.bool_)) or not isinstance(
            confidence,
            Real,
        ):
            raise TypeError("confidence must be numeric and not boolean.")
        if not np.isfinite(confidence):
            raise ValueError("confidence must be finite.")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

        top_k_indices = np.asarray(record["top_k_indices"])
        if top_k_indices.ndim != 1 or top_k_indices.shape[0] == 0:
            raise ValueError("top_k_indices must be a non-empty one-dimensional array.")
        if np.issubdtype(top_k_indices.dtype, np.bool_) or not np.issubdtype(
            top_k_indices.dtype,
            np.integer,
        ):
            raise ValueError("top_k_indices must contain integer labels.")


def per_class_canvas_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k_indices: np.ndarray,
    confidences: np.ndarray,
    num_classes: int = 10,
) -> list[dict[str, float | int]]:
    """
    Compute per-class accuracy, top-k accuracy, error count, and mean confidence.
    """
    _validate_prediction_inputs(y_true, y_pred, num_classes)
    _validate_top_k_indices(top_k_indices, y_true.shape[0], num_classes)
    _validate_confidences(confidences, y_true.shape[0])

    summary = []
    for class_label in range(num_classes):
        class_mask = y_true == class_label
        count = int(np.sum(class_mask))
        if count == 0:
            correct = 0
            errors = 0
            accuracy = float(np.nan)
            top_k_accuracy = float(np.nan)
            mean_confidence = float(np.nan)
            mean_error_confidence = float(np.nan)
        else:
            class_predictions = y_pred[class_mask]
            class_confidences = confidences[class_mask]
            class_top_k = top_k_indices[class_mask]
            correctness = class_predictions == class_label
            correct = int(np.sum(correctness))
            errors = int(count - correct)
            top_k_hits = np.any(class_top_k == class_label, axis=1)
            accuracy = float(correct / count)
            top_k_accuracy = float(np.mean(top_k_hits))
            mean_confidence = float(np.mean(class_confidences))
            mean_error_confidence = (
                float(np.mean(class_confidences[~correctness]))
                if errors > 0
                else float(np.nan)
            )

        summary.append(
            {
                "class_label": int(class_label),
                "count": count,
                "correct": correct,
                "errors": errors,
                "accuracy": accuracy,
                "top_k_accuracy": top_k_accuracy,
                "mean_confidence": mean_confidence,
                "mean_error_confidence": mean_error_confidence,
            }
        )

    return summary


def canvas_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    """
    Compute a confusion matrix for real canvas samples.
    """
    _validate_prediction_inputs(y_true, y_pred, num_classes)

    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def high_confidence_canvas_errors(
    sample_records: list[dict],
    threshold: float = 0.90,
) -> list[dict]:
    """
    Return real canvas errors whose prediction confidence exceeds a threshold.
    """
    _validate_confidence_threshold(threshold)
    _validate_sample_records(sample_records)

    errors = [
        record
        for record in sample_records
        if int(record["prediction"]) != int(record["true_label"])
        and float(record["confidence"]) >= threshold
    ]
    return sorted(errors, key=lambda record: float(record["confidence"]), reverse=True)


def top_k_miss_errors(
    sample_records: list[dict],
) -> list[dict]:
    """
    Return errors where the true label is not present in top-k predictions.
    """
    _validate_sample_records(sample_records)

    errors = []
    for record in sample_records:
        true_label = int(record["true_label"])
        prediction = int(record["prediction"])
        top_k_indices = np.asarray(record["top_k_indices"], dtype=int)
        if prediction != true_label and true_label not in set(top_k_indices.tolist()):
            errors.append(record)

    return sorted(errors, key=lambda record: float(record["confidence"]), reverse=True)


def summarize_canvas_validation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    top_k_indices: np.ndarray,
) -> dict[str, float | int]:
    """
    Summarize real canvas validation performance.
    """
    _validate_probability_inputs(y_true, y_pred, probabilities, top_k_indices)

    confidences = np.max(probabilities, axis=1)
    correctness = y_pred == y_true
    top_k_hits = np.any(top_k_indices == y_true[:, None], axis=1)
    n_samples = int(y_true.shape[0])
    n_errors = int(np.sum(~correctness))
    top_k_miss_error_count = int(np.sum(~correctness & ~top_k_hits))

    accuracy = float(np.mean(correctness))
    top_k_accuracy = float(np.mean(top_k_hits))
    mean_confidence = float(np.mean(confidences))
    mean_confidence_correct = (
        float(np.mean(confidences[correctness]))
        if np.any(correctness)
        else float(np.nan)
    )
    mean_confidence_incorrect = (
        float(np.mean(confidences[~correctness]))
        if n_errors > 0
        else float(np.nan)
    )

    return {
        "n_samples": n_samples,
        "n_errors": n_errors,
        "accuracy": accuracy,
        "top_k_accuracy": top_k_accuracy,
        "mean_confidence": mean_confidence,
        "mean_confidence_correct": mean_confidence_correct,
        "mean_confidence_incorrect": mean_confidence_incorrect,
        "overconfidence_gap": float(mean_confidence - accuracy),
        "top_k_miss_error_count": top_k_miss_error_count,
        "top_k_miss_error_rate": (
            float(top_k_miss_error_count / n_errors) if n_errors > 0 else 0.0
        ),
    }
