import logging

import numpy as np

from src.data.digits import load_digits_dataset
from src.data.preprocessing import stratified_train_val_test_split
from src.models.multiclass_mlp import MulticlassMLPScratch
from src.optimization.parameter_optimizers import ParameterAdam
from src.train import evaluate_multiclass_mlp, train_multiclass_mlp


logger = logging.getLogger("test_multiclass_training")


def _load_digits_split() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X, y, _ = load_digits_dataset()
    return stratified_train_val_test_split(
        X,
        y,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )


def test_evaluate_multiclass_mlp_returns_valid_metrics() -> None:
    X, y, _ = load_digits_dataset()
    X_subset = X[:50]
    y_subset = y[:50]
    model = MulticlassMLPScratch(
        n_features=X_subset.shape[1],
        hidden_dim=16,
        num_classes=10,
        seed=42,
    )

    metrics = evaluate_multiclass_mlp(model, X_subset, y_subset)

    assert set(metrics) == {"cross_entropy", "accuracy"}
    assert np.isfinite(metrics["cross_entropy"])
    assert metrics["cross_entropy"] > 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_train_multiclass_mlp_history_lengths_and_update_count() -> None:
    seed = 42
    num_epochs = 10
    batch_size = 128
    X_train, X_val, _, y_train, y_val, _ = _load_digits_split()
    model = MulticlassMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=16,
        num_classes=10,
        seed=seed,
    )
    optimizer = ParameterAdam(learning_rate=0.01)

    history = train_multiclass_mlp(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
        log_every=10,
        logger=logger,
    )
    expected_updates = num_epochs * int(np.ceil(len(X_train) / batch_size))

    assert len(history["train_cross_entropy"]) == num_epochs
    assert len(history["val_cross_entropy"]) == num_epochs
    assert len(history["train_accuracy"]) == num_epochs
    assert len(history["val_accuracy"]) == num_epochs
    assert history["update_count"] == expected_updates


def test_train_multiclass_mlp_reduces_train_cross_entropy() -> None:
    seed = 42
    X_train, X_val, _, y_train, y_val, _ = _load_digits_split()
    model = MulticlassMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=32,
        num_classes=10,
        seed=seed,
    )
    optimizer = ParameterAdam(learning_rate=0.01)

    initial_train_ce = evaluate_multiclass_mlp(
        model,
        X_train,
        y_train,
    )["cross_entropy"]

    train_multiclass_mlp(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=30,
        batch_size=128,
        seed=seed,
        log_every=30,
        logger=logger,
    )

    final_train_ce = evaluate_multiclass_mlp(
        model,
        X_train,
        y_train,
    )["cross_entropy"]

    assert final_train_ce < initial_train_ce


def test_train_multiclass_mlp_changes_parameters() -> None:
    seed = 42
    X_train, X_val, _, y_train, y_val, _ = _load_digits_split()
    model = MulticlassMLPScratch(
        n_features=X_train.shape[1],
        hidden_dim=16,
        num_classes=10,
        seed=seed,
    )
    optimizer = ParameterAdam(learning_rate=0.01)
    initial_parameters = model.get_parameters()

    train_multiclass_mlp(
        model=model,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=5,
        batch_size=128,
        seed=seed,
        log_every=5,
        logger=logger,
    )

    updated_parameters = model.get_parameters()

    assert any(
        not np.array_equal(initial_parameters[name], updated_parameters[name])
        for name in initial_parameters
    )
