"""Checkpoint utilities for scratch NumPy models."""

import json
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from src.models.multiclass_mlp import MulticlassMLPScratch


CHECKPOINT_VERSION = "1.0"
SUPPORTED_MODEL_CLASS = "MulticlassMLPScratch"
PARAMETER_NAMES = ("W1", "b1", "W2", "b2")


def _validate_required_key(
    metadata: dict[str, Any],
    key: str,
) -> None:
    if key not in metadata:
        raise ValueError(f"metadata must contain {key!r}.")


def _validate_architecture_integer(
    metadata: dict[str, Any],
    key: str,
    minimum: int,
) -> None:
    value = metadata[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"metadata[{key!r}] must be an integer.")
    if value < minimum:
        raise ValueError(f"metadata[{key!r}] must be at least {minimum}.")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")

    required_keys = (
        "checkpoint_version",
        "model_class",
        "n_features",
        "hidden_dim",
        "num_classes",
        "input_scaling",
        "class_names",
    )
    for key in required_keys:
        _validate_required_key(metadata, key)

    if metadata["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version.")
    if metadata["model_class"] != SUPPORTED_MODEL_CLASS:
        raise ValueError("unsupported model class.")

    _validate_architecture_integer(metadata, "n_features", minimum=1)
    _validate_architecture_integer(metadata, "hidden_dim", minimum=1)
    _validate_architecture_integer(metadata, "num_classes", minimum=2)

    if not isinstance(metadata["input_scaling"], str):
        raise TypeError("metadata['input_scaling'] must be a string.")
    if not isinstance(metadata["class_names"], list):
        raise TypeError("metadata['class_names'] must be a list.")
    if not all(isinstance(class_name, str) for class_name in metadata["class_names"]):
        raise TypeError("metadata['class_names'] must contain only strings.")
    if len(metadata["class_names"]) != metadata["num_classes"]:
        raise ValueError("class_names length must equal num_classes.")


def build_multiclass_mlp_metadata(
    model: MulticlassMLPScratch,
    input_scaling: str,
    class_names: list[str],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(model, MulticlassMLPScratch):
        raise TypeError("model must be a MulticlassMLPScratch.")
    if not isinstance(input_scaling, str):
        raise TypeError("input_scaling must be a string.")
    if not isinstance(class_names, list):
        raise TypeError("class_names must be a list.")
    if not all(isinstance(class_name, str) for class_name in class_names):
        raise TypeError("class_names must contain only strings.")

    metadata: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_class": SUPPORTED_MODEL_CLASS,
        "n_features": model.n_features,
        "hidden_dim": model.hidden_dim,
        "num_classes": model.num_classes,
        "input_scaling": input_scaling,
        "class_names": list(class_names),
    }

    if extra_metadata is not None:
        if not isinstance(extra_metadata, dict):
            raise TypeError("extra_metadata must be a dictionary.")
        metadata["extra_metadata"] = dict(extra_metadata)

    _validate_metadata(metadata)
    return metadata


def save_multiclass_mlp_checkpoint(
    model: MulticlassMLPScratch,
    checkpoint_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    if not isinstance(model, MulticlassMLPScratch):
        raise TypeError("model must be a MulticlassMLPScratch.")
    _validate_metadata(metadata)

    if metadata["n_features"] != model.n_features:
        raise ValueError("metadata n_features must match the model.")
    if metadata["hidden_dim"] != model.hidden_dim:
        raise ValueError("metadata hidden_dim must match the model.")
    if metadata["num_classes"] != model.num_classes:
        raise ValueError("metadata num_classes must match the model.")

    parameters = model.get_parameters()
    if set(parameters) != set(PARAMETER_NAMES):
        raise ValueError("model parameters must contain W1, b1, W2, and b2.")

    path = Path(checkpoint_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata),
        W1=parameters["W1"],
        b1=parameters["b1"],
        W2=parameters["W2"],
        b2=parameters["b2"],
    )


def load_multiclass_mlp_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[MulticlassMLPScratch, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    required_keys = {"metadata_json", *PARAMETER_NAMES}
    with np.load(path, allow_pickle=False) as data:
        missing_keys = required_keys - set(data.files)
        if missing_keys:
            raise ValueError(f"checkpoint is missing keys: {sorted(missing_keys)}")

        metadata_json_array = data["metadata_json"]
        metadata_json = (
            metadata_json_array.item()
            if metadata_json_array.shape == ()
            else str(metadata_json_array)
        )
        metadata = json.loads(metadata_json)
        _validate_metadata(metadata)

        parameters = {
            name: data[name].copy()
            for name in PARAMETER_NAMES
        }

    model = MulticlassMLPScratch(
        n_features=metadata["n_features"],
        hidden_dim=metadata["hidden_dim"],
        num_classes=metadata["num_classes"],
        seed=0,
    )
    model.set_parameters(parameters)

    return model, metadata
