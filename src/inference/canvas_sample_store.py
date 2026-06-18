"""Utilities for saving and loading user-drawn canvas digit samples."""

import time
from pathlib import Path
from typing import Any

import numpy as np


def validate_optional_digit_label(label: int | None) -> None:
    if label is None:
        return
    if isinstance(label, (bool, np.bool_)) or not isinstance(
        label,
        (int, np.integer),
    ):
        raise TypeError("label must be an integer or None.")
    if label < 0 or label > 9:
        raise ValueError("label must satisfy 0 <= label <= 9.")


def _validate_finite_array(
    name: str,
    array: np.ndarray,
) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")


def _require_stage_array(
    stages: dict[str, Any],
    key: str,
) -> np.ndarray:
    if key not in stages:
        raise ValueError(f"preprocessing_stages must contain {key!r}.")
    array = np.asarray(stages[key], dtype=float)
    _validate_finite_array(key, array)
    return array


def _encode_bounding_box(
    bounding_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if bounding_box is None:
        return np.array([-1, -1, -1, -1], dtype=int)
    if len(bounding_box) != 4:
        raise ValueError("bounding_box must contain four values.")
    return np.asarray(bounding_box, dtype=int)


def _unique_sample_path(
    output_dir: Path,
    base_name: str,
) -> Path:
    candidate = output_dir / f"{base_name}.npz"
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = output_dir / f"{base_name}_{suffix:02d}.npz"
        if not candidate.exists():
            return candidate
        suffix += 1


def save_canvas_sample(
    output_dir: str | Path,
    raw_canvas: np.ndarray,
    preprocessing_stages: dict[str, Any],
    prediction_result: dict[str, Any],
    checkpoint_path: str | Path,
    true_label: int | None = None,
) -> Path:
    """
    Save one user-drawn canvas sample and metadata as a compressed npz file.
    """
    validate_optional_digit_label(true_label)
    _validate_finite_array("raw_canvas", raw_canvas)
    if not isinstance(preprocessing_stages, dict):
        raise TypeError("preprocessing_stages must be a dictionary.")
    if not isinstance(prediction_result, dict):
        raise TypeError("prediction_result must be a dictionary.")

    normalized = _require_stage_array(preprocessing_stages, "normalized")
    bright_foreground = _require_stage_array(
        preprocessing_stages,
        "bright_foreground",
    )
    cropped = _require_stage_array(preprocessing_stages, "cropped")
    resized_8x8 = _require_stage_array(preprocessing_stages, "resized_8x8")
    feature_vector = _require_stage_array(preprocessing_stages, "feature_vector")

    if resized_8x8.shape != (8, 8):
        raise ValueError("resized_8x8 must have shape (8, 8).")
    if feature_vector.shape != (64,):
        raise ValueError("feature_vector must have shape (64,).")

    probabilities = np.asarray(prediction_result["probabilities"], dtype=float)
    top_k_indices = np.asarray(prediction_result["top_k_indices"], dtype=int)
    top_k_probabilities = np.asarray(
        prediction_result["top_k_probabilities"],
        dtype=float,
    )
    _validate_finite_array("probabilities", probabilities)
    _validate_finite_array("top_k_probabilities", top_k_probabilities)

    prediction = int(prediction_result["prediction"])
    confidence = float(prediction_result["confidence"])
    if not np.isfinite(confidence):
        raise ValueError("confidence must be finite.")

    is_blank = bool(preprocessing_stages["is_blank"])
    foreground_mass = float(preprocessing_stages["foreground_mass"])
    if not np.isfinite(foreground_mass):
        raise ValueError("foreground_mass must be finite.")
    bounding_box = _encode_bounding_box(preprocessing_stages["bounding_box"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    label_text = f"label_{true_label}" if true_label is not None else "unlabeled"
    sample_path = _unique_sample_path(
        output_path,
        f"sample_{timestamp}_{label_text}",
    )
    np.savez_compressed(
        sample_path,
        raw_canvas=np.asarray(raw_canvas, dtype=float),
        normalized=normalized,
        bright_foreground=bright_foreground,
        cropped=cropped,
        resized_8x8=resized_8x8,
        feature_vector=feature_vector,
        probabilities=probabilities,
        top_k_indices=top_k_indices,
        top_k_probabilities=top_k_probabilities,
        prediction=np.array(prediction, dtype=int),
        confidence=np.array(confidence, dtype=float),
        true_label=np.array(-1 if true_label is None else true_label, dtype=int),
        checkpoint_path=np.array(str(checkpoint_path)),
        timestamp=np.array(timestamp),
        is_blank=np.array(is_blank),
        foreground_mass=np.array(foreground_mass, dtype=float),
        bounding_box=bounding_box,
    )

    return sample_path


def _scalar_string(array: np.ndarray) -> str:
    return str(array.item() if array.shape == () else array)


def load_canvas_samples(
    samples_dir: str | Path,
    require_labels: bool = True,
) -> list[dict[str, Any]]:
    """
    Load saved canvas samples.
    """
    directory = Path(samples_dir)
    if not directory.exists():
        return []

    samples = []
    for sample_path in sorted(directory.glob("*.npz")):
        with np.load(sample_path, allow_pickle=False) as data:
            true_label = int(data["true_label"].item())
            if require_labels and true_label == -1:
                continue

            sample = {
                "path": sample_path,
                "raw_canvas": data["raw_canvas"].copy(),
                "resized_8x8": data["resized_8x8"].copy(),
                "feature_vector": data["feature_vector"].copy(),
                "true_label": true_label,
                "saved_prediction": int(data["prediction"].item()),
                "saved_confidence": float(data["confidence"].item()),
                "checkpoint_path": _scalar_string(data["checkpoint_path"]),
                "foreground_mass": float(data["foreground_mass"].item()),
                "is_blank": bool(data["is_blank"].item()),
            }
            if "probabilities" in data.files:
                sample["probabilities"] = data["probabilities"].copy()
            samples.append(sample)

    return samples


def stack_canvas_sample_features(
    samples: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack labeled canvas sample feature vectors and labels.
    """
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("samples must be a non-empty list.")

    features = []
    labels = []
    for sample in samples:
        label = int(sample["true_label"])
        if label == -1:
            raise ValueError("all samples must be labeled.")
        feature_vector = np.asarray(sample["feature_vector"], dtype=float)
        if feature_vector.shape != (64,):
            raise ValueError("feature_vector must have shape (64,).")
        features.append(feature_vector)
        labels.append(label)

    return np.vstack(features), np.asarray(labels, dtype=int)
