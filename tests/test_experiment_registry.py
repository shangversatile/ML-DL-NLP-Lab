import json

import numpy as np
import pytest

from src.evaluation.experiment_registry import (
    append_experiment_record,
    load_experiment_records,
    make_experiment_record,
    summarize_records,
    validate_metric_value,
    validate_metrics,
)


def test_validate_metric_value_accepts_json_scalar_values() -> None:
    assert validate_metric_value(3) == 3
    assert validate_metric_value(0.75) == pytest.approx(0.75)
    assert validate_metric_value("validation") == "validation"
    assert validate_metric_value(True) is True
    assert validate_metric_value(None) is None


def test_validate_metric_value_accepts_numpy_scalar_values() -> None:
    assert validate_metric_value(np.int64(7)) == 7
    assert validate_metric_value(np.float32(0.5)) == pytest.approx(0.5)
    assert validate_metric_value(np.bool_(True)) is True
    assert validate_metric_value(np.str_("digits")) == "digits"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_metric_value_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_metric_value(value)


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        {"accuracy": 0.9},
        np.array([1, 2, 3]),
        object(),
    ],
)
def test_validate_metric_value_rejects_container_and_object_values(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="Metric values"):
        validate_metric_value(value)


def test_validate_metrics_accepts_valid_metric_dict() -> None:
    metrics = validate_metrics(
        {
            "accuracy": np.float64(0.9),
            "n_samples": np.int64(100),
            "split_role": "validation",
            "used_abstention": False,
            "optional_note": None,
        }
    )

    assert metrics == {
        "accuracy": pytest.approx(0.9),
        "n_samples": 100,
        "split_role": "validation",
        "used_abstention": False,
        "optional_note": None,
    }


def test_validate_metrics_rejects_empty_dict() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_metrics({})


def test_validate_metrics_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError, match="keys"):
        validate_metrics({1: 0.9})  # type: ignore[arg-type]


def test_validate_metrics_rejects_empty_string_keys() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_metrics({"": 0.9})


def test_validate_metrics_rejects_invalid_metric_values() -> None:
    with pytest.raises(TypeError, match="Metric values"):
        validate_metrics({"bad": [1, 2, 3]})


def test_make_experiment_record_creates_required_fields() -> None:
    record = make_experiment_record(
        name="digits_baseline_eval",
        split="validation",
        metrics={"accuracy": np.float32(0.92)},
        model="MulticlassMLPScratch",
        dataset="load_digits",
        checkpoint="results/checkpoints/digits_mlp.npz",
        notes="diagnostic baseline",
        tags=["week5", "registry"],
    )

    assert record["name"] == "digits_baseline_eval"
    assert record["split"] == "validation"
    assert record["metrics"] == {"accuracy": pytest.approx(0.92)}
    assert record["model"] == "MulticlassMLPScratch"
    assert record["dataset"] == "load_digits"
    assert record["checkpoint"] == "results/checkpoints/digits_mlp.npz"
    assert record["notes"] == "diagnostic baseline"
    assert record["tags"] == ["week5", "registry"]
    assert isinstance(record["created_at_unix"], float)


def test_make_experiment_record_validates_metrics() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_experiment_record(
            name="bad_metric",
            split="validation",
            metrics={"accuracy": float("nan")},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "split": "validation"},
        {"name": "digits", "split": ""},
    ],
)
def test_make_experiment_record_validates_non_empty_name_and_split(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_experiment_record(metrics={"accuracy": 0.9}, **kwargs)


@pytest.mark.parametrize(
    "tags,expected_error",
    [
        ("week5", TypeError),
        ([1], TypeError),
        ([""], ValueError),
    ],
)
def test_make_experiment_record_validates_tags(
    tags: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        make_experiment_record(
            name="digits",
            split="validation",
            metrics={"accuracy": 0.9},
            tags=tags,  # type: ignore[arg-type]
        )


def test_append_and_load_experiment_record_round_trip(tmp_path) -> None:
    registry_path = tmp_path / "registries" / "experiments.jsonl"
    record = make_experiment_record(
        name="digits_baseline_eval",
        split="validation",
        metrics={"accuracy": 0.9},
        dataset="load_digits",
        tags=["baseline"],
    )

    output_path = append_experiment_record(record, registry_path)
    loaded_records = load_experiment_records(registry_path)

    assert output_path == registry_path
    assert loaded_records == [record]


def test_append_and_load_multiple_experiment_records(tmp_path) -> None:
    registry_path = tmp_path / "experiments.jsonl"
    first = make_experiment_record(
        name="baseline",
        split="validation",
        metrics={"accuracy": 0.9},
    )
    second = make_experiment_record(
        name="robustness_probe",
        split="diagnostic",
        metrics={"accuracy": 0.4},
    )

    append_experiment_record(first, registry_path)
    append_experiment_record(second, registry_path)

    assert load_experiment_records(registry_path) == [first, second]


def test_load_experiment_records_missing_file_returns_empty_list(tmp_path) -> None:
    assert load_experiment_records(tmp_path / "missing.jsonl") == []


def test_load_experiment_records_empty_file_returns_empty_list(tmp_path) -> None:
    registry_path = tmp_path / "empty.jsonl"
    registry_path.write_text("", encoding="utf-8")

    assert load_experiment_records(registry_path) == []


def test_load_experiment_records_invalid_jsonl_line_raises_clear_error(
    tmp_path,
) -> None:
    registry_path = tmp_path / "bad.jsonl"
    registry_path.write_text(
        json.dumps({"name": "valid"}) + "\n{bad json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON on line 2"):
        load_experiment_records(registry_path)


def test_load_experiment_records_rejects_non_object_lines(tmp_path) -> None:
    registry_path = tmp_path / "bad.jsonl"
    registry_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_experiment_records(registry_path)


def test_summarize_records_summarizes_registry_fields() -> None:
    records = [
        {
            "name": "baseline",
            "split": "validation",
            "dataset": "load_digits",
            "model": "MulticlassMLPScratch",
            "tags": ["week5", "baseline"],
        },
        {
            "name": "canvas_probe",
            "split": "diagnostic",
            "dataset": "Canvas-Diagnostic-v1",
            "model": "MulticlassMLPScratch",
            "tags": ["week5", "canvas"],
        },
        {
            "name": "baseline",
            "split": "test",
            "dataset": None,
            "tags": [],
        },
    ]

    summary = summarize_records(records)

    assert summary == {
        "n_records": 3,
        "names": ["baseline", "canvas_probe"],
        "datasets": ["Canvas-Diagnostic-v1", "load_digits"],
        "splits": ["diagnostic", "test", "validation"],
        "models": ["MulticlassMLPScratch"],
        "tags": ["baseline", "canvas", "week5"],
    }


def test_summarize_records_handles_empty_input() -> None:
    assert summarize_records([]) == {
        "n_records": 0,
        "names": [],
        "datasets": [],
        "splits": [],
        "models": [],
        "tags": [],
    }
