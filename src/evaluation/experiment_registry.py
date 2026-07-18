"""Utilities for recording and validating experiment evaluation results."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


MetricValue = float | int | str | bool | None


def validate_metric_value(value: Any) -> MetricValue:
    """
    Validate and normalize a metric value so it can be safely serialized.
    """
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Metric float values must be finite.")
        return value

    if isinstance(value, str):
        return value

    if value is None:
        return None

    raise TypeError(
        "Metric values must be int, float, str, bool, None, or NumPy scalars."
    )


def validate_metrics(metrics: dict[str, Any]) -> dict[str, MetricValue]:
    """
    Validate a metrics dictionary and return a JSON-serializable copy.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary.")

    if not metrics:
        raise ValueError("metrics must be non-empty.")

    validated: dict[str, MetricValue] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError("Metric keys must be strings.")
        if not key.strip():
            raise ValueError("Metric keys must be non-empty strings.")
        validated[key] = validate_metric_value(value)

    return validated


def _validate_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _validate_optional_string(value: Any, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None.")
    return value


def _validate_tags(tags: list[str] | None) -> list[str]:
    if tags is None:
        return []
    if not isinstance(tags, list):
        raise TypeError("tags must be a list of strings or None.")

    validated_tags: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise TypeError("tags must contain only strings.")
        if not tag.strip():
            raise ValueError("tags must contain only non-empty strings.")
        validated_tags.append(tag)
    return validated_tags


def make_experiment_record(
    *,
    name: str,
    split: str,
    metrics: dict[str, Any],
    model: str | None = None,
    dataset: str | None = None,
    checkpoint: str | None = None,
    notes: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a standardized experiment record.
    """
    return {
        "name": _validate_non_empty_string(name, "name"),
        "split": _validate_non_empty_string(split, "split"),
        "metrics": validate_metrics(metrics),
        "model": _validate_optional_string(model, "model"),
        "dataset": _validate_optional_string(dataset, "dataset"),
        "checkpoint": _validate_optional_string(checkpoint, "checkpoint"),
        "notes": _validate_optional_string(notes, "notes"),
        "tags": _validate_tags(tags),
        "created_at_unix": time.time(),
    }


def append_experiment_record(record: dict[str, Any], output_path: str | Path) -> Path:
    """
    Append one experiment record to a JSONL registry file.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    return path


def load_experiment_records(path: str | Path) -> list[dict[str, Any]]:
    """
    Load experiment records from a JSONL registry file.
    """
    registry_path = Path(path)
    if not registry_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with registry_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {registry_path}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} in {registry_path} must contain a JSON object."
                )
            records.append(record)

    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize a list of experiment records.
    """
    names: set[str] = set()
    datasets: set[str] = set()
    splits: set[str] = set()
    models: set[str] = set()
    tags: set[str] = set()

    for record in records:
        name = record.get("name")
        if isinstance(name, str) and name:
            names.add(name)

        dataset = record.get("dataset")
        if isinstance(dataset, str) and dataset:
            datasets.add(dataset)

        split = record.get("split")
        if isinstance(split, str) and split:
            splits.add(split)

        model = record.get("model")
        if isinstance(model, str) and model:
            models.add(model)

        record_tags = record.get("tags", [])
        if isinstance(record_tags, list):
            tags.update(tag for tag in record_tags if isinstance(tag, str) and tag)

    return {
        "n_records": len(records),
        "names": sorted(names),
        "datasets": sorted(datasets),
        "splits": sorted(splits),
        "models": sorted(models),
        "tags": sorted(tags),
    }
