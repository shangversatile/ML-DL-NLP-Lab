import numpy as np
import pytest

from src.models.checkpoint import (
    CHECKPOINT_VERSION,
    SUPPORTED_MODEL_CLASS,
    _validate_metadata,
    build_multiclass_mlp_metadata,
    load_multiclass_mlp_checkpoint,
    save_multiclass_mlp_checkpoint,
)
from src.models.multiclass_mlp import MulticlassMLPScratch


def _create_model() -> MulticlassMLPScratch:
    return MulticlassMLPScratch(
        n_features=4,
        hidden_dim=5,
        num_classes=3,
        seed=42,
    )


def _create_metadata(
    model: MulticlassMLPScratch,
) -> dict:
    return build_multiclass_mlp_metadata(
        model=model,
        input_scaling="unit-test scaling",
        class_names=["zero", "one", "two"],
        extra_metadata={"seed": 42},
    )


def test_build_multiclass_mlp_metadata() -> None:
    model = _create_model()

    metadata = _create_metadata(model)

    assert metadata["checkpoint_version"] == CHECKPOINT_VERSION
    assert metadata["model_class"] == SUPPORTED_MODEL_CLASS
    assert metadata["n_features"] == model.n_features
    assert metadata["hidden_dim"] == model.hidden_dim
    assert metadata["num_classes"] == model.num_classes
    assert metadata["input_scaling"] == "unit-test scaling"
    assert metadata["class_names"] == ["zero", "one", "two"]
    assert metadata["extra_metadata"] == {"seed": 42}


def test_save_and_load_round_trip_preserves_predictions(tmp_path) -> None:
    model = _create_model()
    metadata = _create_metadata(model)
    X = np.array(
        [
            [0.0, 0.25, 0.5, 0.75],
            [1.0, 0.5, 0.25, 0.0],
        ]
    )
    original_probabilities = model.predict_proba(X)
    checkpoint_path = tmp_path / "model.npz"

    save_multiclass_mlp_checkpoint(model, checkpoint_path, metadata)
    loaded_model, loaded_metadata = load_multiclass_mlp_checkpoint(checkpoint_path)
    loaded_probabilities = loaded_model.predict_proba(X)

    np.testing.assert_allclose(loaded_probabilities, original_probabilities)
    np.testing.assert_array_equal(loaded_model.predict(X), model.predict(X))
    assert loaded_metadata == metadata


def test_save_creates_parent_directory(tmp_path) -> None:
    model = _create_model()
    metadata = _create_metadata(model)
    checkpoint_path = tmp_path / "nested" / "checkpoints" / "model.npz"

    save_multiclass_mlp_checkpoint(model, checkpoint_path, metadata)

    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0


def test_loaded_model_has_correct_architecture(tmp_path) -> None:
    model = _create_model()
    metadata = _create_metadata(model)
    checkpoint_path = tmp_path / "model.npz"

    save_multiclass_mlp_checkpoint(model, checkpoint_path, metadata)
    loaded_model, _ = load_multiclass_mlp_checkpoint(checkpoint_path)

    assert loaded_model.n_features == model.n_features
    assert loaded_model.hidden_dim == model.hidden_dim
    assert loaded_model.num_classes == model.num_classes


def test_validate_metadata_rejects_missing_required_key() -> None:
    metadata = _create_metadata(_create_model())
    metadata.pop("checkpoint_version")

    with pytest.raises(ValueError):
        _validate_metadata(metadata)


def test_validate_metadata_rejects_wrong_checkpoint_version() -> None:
    metadata = _create_metadata(_create_model())
    metadata["checkpoint_version"] = "0.0"

    with pytest.raises(ValueError):
        _validate_metadata(metadata)


def test_validate_metadata_rejects_wrong_model_class() -> None:
    metadata = _create_metadata(_create_model())
    metadata["model_class"] = "OtherModel"

    with pytest.raises(ValueError):
        _validate_metadata(metadata)


@pytest.mark.parametrize(
    "key,value,expected_error",
    [
        ("n_features", 0, ValueError),
        ("hidden_dim", -1, ValueError),
        ("num_classes", 1, ValueError),
        ("n_features", True, TypeError),
        ("hidden_dim", 5.0, TypeError),
    ],
)
def test_validate_metadata_rejects_invalid_architecture_values(
    key: str,
    value: object,
    expected_error: type[Exception],
) -> None:
    metadata = _create_metadata(_create_model())
    metadata[key] = value

    with pytest.raises(expected_error):
        _validate_metadata(metadata)


def test_validate_metadata_rejects_class_name_length_mismatch() -> None:
    metadata = _create_metadata(_create_model())
    metadata["class_names"] = ["zero", "one"]

    with pytest.raises(ValueError):
        _validate_metadata(metadata)


def test_validate_metadata_rejects_non_string_class_names() -> None:
    metadata = _create_metadata(_create_model())
    metadata["class_names"] = ["zero", 1, "two"]

    with pytest.raises(TypeError):
        _validate_metadata(metadata)


def test_load_missing_checkpoint_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_multiclass_mlp_checkpoint(tmp_path / "missing.npz")


def test_save_rejects_metadata_model_mismatch(tmp_path) -> None:
    model = _create_model()
    metadata = _create_metadata(model)
    metadata["hidden_dim"] = model.hidden_dim + 1

    with pytest.raises(ValueError):
        save_multiclass_mlp_checkpoint(model, tmp_path / "model.npz", metadata)
