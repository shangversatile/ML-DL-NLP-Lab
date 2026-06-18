"""Tests for fixed-update multiclass MLP training."""

import logging

import numpy as np
import pytest

from src.data.digits import load_digits_dataset
from src.models.multiclass_mlp import MulticlassMLPScratch
from src.optimization.parameter_optimizers import ParameterAdam
from src.train import (
    evaluate_multiclass_mlp,
    train_multiclass_mlp_fixed_updates,
)


logger = logging.getLogger("test_multiclass_fixed_update_training")


def _small_digits_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y, _ = load_digits_dataset(scale_pixels=True)
    train_indices = np.arange(120)
    val_indices = np.arange(120, 180)
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def _model_and_optimizer(seed: int = 0) -> tuple[MulticlassMLPScratch, ParameterAdam]:
    model = MulticlassMLPScratch(
        n_features=64,
        hidden_dim=16,
        num_classes=10,
        seed=seed,
    )
    optimizer = ParameterAdam(learning_rate=0.005)
    return model, optimizer


def test_fixed_update_count_is_exact() -> None:
    X_train, y_train, X_val, y_val = _small_digits_split()
    model, optimizer = _model_and_optimizer(seed=1)

    history = train_multiclass_mlp_fixed_updates(
        model,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        max_updates=7,
        batch_size=32,
        seed=42,
        eval_every_updates=3,
        log_every_updates=20,
        logger=logger,
    )

    assert history["update_count"] == 7


def test_initial_intermediate_and_final_evaluations_are_recorded() -> None:
    X_train, y_train, X_val, y_val = _small_digits_split()
    model, optimizer = _model_and_optimizer(seed=2)

    history = train_multiclass_mlp_fixed_updates(
        model,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        max_updates=7,
        batch_size=32,
        seed=42,
        eval_every_updates=3,
        log_every_updates=20,
        logger=logger,
    )

    assert history["update_steps"] == [0, 3, 6, 7]
    assert len(history["train_cross_entropy"]) == 4
    assert len(history["val_accuracy"]) == 4


def test_parameters_change_after_fixed_update_training() -> None:
    X_train, y_train, X_val, y_val = _small_digits_split()
    model, optimizer = _model_and_optimizer(seed=3)
    before = model.get_parameters()

    train_multiclass_mlp_fixed_updates(
        model,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        max_updates=5,
        batch_size=32,
        seed=42,
        eval_every_updates=5,
        log_every_updates=20,
        logger=logger,
    )

    after = model.get_parameters()
    assert any(not np.allclose(before[name], after[name]) for name in before)


def test_fixed_update_training_reduces_or_keeps_finite_train_ce() -> None:
    X_train, y_train, X_val, y_val = _small_digits_split()
    model, optimizer = _model_and_optimizer(seed=4)
    initial_ce = evaluate_multiclass_mlp(model, X_train, y_train)["cross_entropy"]

    history = train_multiclass_mlp_fixed_updates(
        model,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        max_updates=12,
        batch_size=32,
        seed=42,
        eval_every_updates=6,
        log_every_updates=20,
        logger=logger,
    )

    final_ce = history["train_cross_entropy"][-1]
    assert np.isfinite(final_ce)
    assert final_ce <= initial_ce


@pytest.mark.parametrize(
    "argument, value, expected_error",
    [
        ("max_updates", 0, ValueError),
        ("max_updates", True, TypeError),
        ("batch_size", 0, ValueError),
        ("batch_size", True, TypeError),
        ("seed", 1.5, TypeError),
        ("seed", True, TypeError),
        ("eval_every_updates", 0, ValueError),
        ("eval_every_updates", True, TypeError),
        ("log_every_updates", 0, ValueError),
        ("log_every_updates", True, TypeError),
    ],
)
def test_fixed_update_training_rejects_invalid_arguments(
    argument: str,
    value,
    expected_error: type[Exception],
) -> None:
    X_train, y_train, X_val, y_val = _small_digits_split()
    model, optimizer = _model_and_optimizer(seed=5)
    kwargs = {
        "max_updates": 5,
        "batch_size": 32,
        "seed": 42,
        "eval_every_updates": 2,
        "log_every_updates": 4,
    }
    kwargs[argument] = value

    with pytest.raises(expected_error):
        train_multiclass_mlp_fixed_updates(
            model,
            optimizer,
            X_train,
            y_train,
            X_val,
            y_val,
            logger=logger,
            **kwargs,
        )
