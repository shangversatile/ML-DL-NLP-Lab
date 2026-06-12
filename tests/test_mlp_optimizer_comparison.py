import logging

import numpy as np
import pytest

from experiments.compare_mlp_optimizers import (
    create_optimizer,
    run_optimizer_experiment,
)
from src.data.datasets import make_xor_classification_data
from src.data.preprocessing import standardize_features, train_val_split
from src.models.mlp import BinaryMLPScratch
from src.optimization.parameter_optimizers import (
    ParameterAdam,
    ParameterMomentum,
    ParameterSGD,
)


logger = logging.getLogger("test_mlp_optimizer_comparison")


def prepare_small_comparison_case():
    X, y = make_xor_classification_data(
        n_samples=120,
        noise=0.10,
        seed=42,
    )

    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y,
        val_ratio=0.2,
        seed=42,
    )

    X_train_scaled, X_val_scaled, _, _ = standardize_features(
        X_train,
        X_val,
    )

    base_model = BinaryMLPScratch(
        n_features=2,
        hidden_dim=8,
        seed=42,
    )

    initial_parameters = base_model.get_parameters()

    return (
        initial_parameters,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
    )


def test_create_optimizer_returns_expected_optimizer_types() -> None:
    assert isinstance(
        create_optimizer("sgd", {"learning_rate": 0.05}),
        ParameterSGD,
    )
    assert isinstance(
        create_optimizer("momentum", {"learning_rate": 0.05, "beta": 0.9}),
        ParameterMomentum,
    )
    assert isinstance(
        create_optimizer(
            "adam",
            {
                "learning_rate": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
            },
        ),
        ParameterAdam,
    )

    with pytest.raises(ValueError):
        create_optimizer("unknown", {"learning_rate": 0.01})


def test_shared_initial_parameters_are_not_mutated() -> None:
    (
        initial_parameters,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
    ) = prepare_small_comparison_case()
    initial_parameter_copies = {
        name: value.copy()
        for name, value in initial_parameters.items()
    }

    run_optimizer_experiment(
        optimizer_name="sgd",
        optimizer_config={"learning_rate": 0.05},
        initial_parameters=initial_parameters,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        hidden_dim=8,
        seed=42,
        num_epochs=3,
        batch_size=16,
        log_every=10,
        threshold=0.5,
        logger=logger,
    )

    for name in initial_parameters:
        assert np.array_equal(initial_parameters[name], initial_parameter_copies[name])


def test_optimizer_runs_have_equal_update_budgets() -> None:
    (
        initial_parameters,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
    ) = prepare_small_comparison_case()
    optimizer_configs = {
        "sgd": {"learning_rate": 0.05},
        "momentum": {"learning_rate": 0.05, "beta": 0.9},
        "adam": {
            "learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        },
    }
    results = {
        name: run_optimizer_experiment(
            optimizer_name=name,
            optimizer_config=optimizer_configs[name],
            initial_parameters=initial_parameters,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            hidden_dim=8,
            seed=42,
            num_epochs=5,
            batch_size=16,
            log_every=10,
            threshold=0.5,
            logger=logger,
        )
        for name in optimizer_configs
    }

    update_counts = [
        result["history"]["update_count"]
        for result in results.values()
    ]
    expected_update_count = 5 * int(np.ceil(len(X_train_scaled) / 16))

    assert len(set(update_counts)) == 1
    assert update_counts[0] == expected_update_count


def test_histories_have_equal_lengths() -> None:
    (
        initial_parameters,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
    ) = prepare_small_comparison_case()
    num_epochs = 5
    optimizer_configs = {
        "sgd": {"learning_rate": 0.05},
        "momentum": {"learning_rate": 0.05, "beta": 0.9},
        "adam": {
            "learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        },
    }
    results = [
        run_optimizer_experiment(
            optimizer_name=name,
            optimizer_config=config,
            initial_parameters=initial_parameters,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            hidden_dim=8,
            seed=42,
            num_epochs=num_epochs,
            batch_size=16,
            log_every=10,
            threshold=0.5,
            logger=logger,
        )
        for name, config in optimizer_configs.items()
    ]

    for result in results:
        assert len(result["history"]["train_bce"]) == num_epochs
        assert len(result["history"]["val_bce"]) == num_epochs
        assert len(result["history"]["val_accuracy"]) == num_epochs


def test_all_models_begin_with_identical_initial_metrics() -> None:
    (
        initial_parameters,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
    ) = prepare_small_comparison_case()
    optimizer_configs = {
        "sgd": {"learning_rate": 0.05},
        "momentum": {"learning_rate": 0.05, "beta": 0.9},
        "adam": {
            "learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        },
    }
    results = [
        run_optimizer_experiment(
            optimizer_name=name,
            optimizer_config=config,
            initial_parameters=initial_parameters,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            hidden_dim=8,
            seed=42,
            num_epochs=5,
            batch_size=16,
            log_every=10,
            threshold=0.5,
            logger=logger,
        )
        for name, config in optimizer_configs.items()
    ]

    initial_train_bces = [
        result["initial_train_metrics"]["bce"]
        for result in results
    ]
    initial_val_bces = [
        result["initial_val_metrics"]["bce"]
        for result in results
    ]
    initial_train_accuracies = [
        result["initial_train_metrics"]["accuracy"]
        for result in results
    ]
    initial_val_accuracies = [
        result["initial_val_metrics"]["accuracy"]
        for result in results
    ]

    assert np.allclose(initial_train_bces, initial_train_bces[0])
    assert np.allclose(initial_val_bces, initial_val_bces[0])
    assert np.allclose(initial_train_accuracies, initial_train_accuracies[0])
    assert np.allclose(initial_val_accuracies, initial_val_accuracies[0])


def test_final_parameters_change_for_each_optimizer() -> None:
    (
        initial_parameters,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
    ) = prepare_small_comparison_case()
    optimizer_configs = {
        "sgd": {"learning_rate": 0.05},
        "momentum": {"learning_rate": 0.05, "beta": 0.9},
        "adam": {
            "learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        },
    }

    for name, config in optimizer_configs.items():
        result = run_optimizer_experiment(
            optimizer_name=name,
            optimizer_config=config,
            initial_parameters=initial_parameters,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            hidden_dim=8,
            seed=42,
            num_epochs=5,
            batch_size=16,
            log_every=10,
            threshold=0.5,
            logger=logger,
        )

        assert any(
            not np.array_equal(
                initial_parameters[parameter_name],
                result["final_parameters"][parameter_name],
            )
            for parameter_name in initial_parameters
        )
