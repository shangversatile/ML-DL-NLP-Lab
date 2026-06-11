import numpy as np
import pytest

from src.optimization.parameter_optimizers import (
    ParameterAdam,
    ParameterMomentum,
    ParameterSGD,
)


def create_parameter_case():
    parameters = {
        "W1": np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
        "b1": np.array([0.5, -0.5]),
    }

    gradients = {
        "W1": np.array(
            [
                [0.1, -0.2],
                [0.3, -0.4],
            ]
        ),
        "b1": np.array([0.2, -0.1]),
    }

    return parameters, gradients


def copy_arrays(values):
    return {
        name: value.copy()
        for name, value in values.items()
    }


def assert_dict_arrays_unchanged(actual, expected):
    for name in expected:
        assert np.array_equal(actual[name], expected[name])


def test_parameter_sgd_exact_update_does_not_mutate_inputs() -> None:
    parameters, gradients = create_parameter_case()
    original_parameters = copy_arrays(parameters)
    original_gradients = copy_arrays(gradients)
    optimizer = ParameterSGD(learning_rate=0.1)

    updated_parameters = optimizer.step(parameters, gradients)

    expected_W1 = np.array(
        [
            [0.99, 2.02],
            [2.97, 4.04],
        ]
    )
    expected_b1 = np.array([0.48, -0.49])

    assert np.allclose(updated_parameters["W1"], expected_W1)
    assert np.allclose(updated_parameters["b1"], expected_b1)
    assert_dict_arrays_unchanged(parameters, original_parameters)
    assert_dict_arrays_unchanged(gradients, original_gradients)


def test_parameter_momentum_first_step() -> None:
    parameters, gradients = create_parameter_case()
    optimizer = ParameterMomentum(learning_rate=0.1, beta=0.9)

    updated_parameters = optimizer.step(parameters, gradients)

    expected_velocity_W1 = np.array(
        [
            [0.01, -0.02],
            [0.03, -0.04],
        ]
    )
    expected_velocity_b1 = np.array([0.02, -0.01])
    expected_updated_W1 = parameters["W1"] - 0.1 * expected_velocity_W1
    expected_updated_b1 = parameters["b1"] - 0.1 * expected_velocity_b1

    assert np.allclose(optimizer.velocity["W1"], expected_velocity_W1)
    assert np.allclose(optimizer.velocity["b1"], expected_velocity_b1)
    assert np.allclose(updated_parameters["W1"], expected_updated_W1)
    assert np.allclose(updated_parameters["b1"], expected_updated_b1)


def test_parameter_momentum_second_step_accumulates_state() -> None:
    parameters, gradients = create_parameter_case()
    optimizer = ParameterMomentum(learning_rate=0.1, beta=0.9)

    optimizer.step(parameters, gradients)
    updated_parameters = optimizer.step(parameters, gradients)

    expected_velocity_W1 = np.array(
        [
            [0.019, -0.038],
            [0.057, -0.076],
        ]
    )
    expected_velocity_b1 = np.array([0.038, -0.019])

    assert set(optimizer.velocity) == set(parameters)
    for name in parameters:
        assert optimizer.velocity[name].shape == parameters[name].shape

    assert np.allclose(optimizer.velocity["W1"], expected_velocity_W1)
    assert np.allclose(optimizer.velocity["b1"], expected_velocity_b1)
    assert np.allclose(
        updated_parameters["W1"],
        parameters["W1"] - 0.1 * expected_velocity_W1,
    )
    assert np.allclose(
        updated_parameters["b1"],
        parameters["b1"] - 0.1 * expected_velocity_b1,
    )


def test_parameter_momentum_rejects_incompatible_state() -> None:
    parameters, gradients = create_parameter_case()
    optimizer = ParameterMomentum(learning_rate=0.1, beta=0.9)
    optimizer.step(parameters, gradients)
    optimizer.velocity["W1"] = np.zeros(3)

    with pytest.raises(ValueError):
        optimizer.step(parameters, gradients)


def test_parameter_adam_initialization() -> None:
    optimizer = ParameterAdam()

    assert optimizer.first_moment == {}
    assert optimizer.second_moment == {}
    assert optimizer.time_step == 0


def test_parameter_adam_first_step() -> None:
    parameters, gradients = create_parameter_case()
    original_parameters = copy_arrays(parameters)
    optimizer = ParameterAdam(
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )

    updated_parameters = optimizer.step(parameters, gradients)

    assert optimizer.time_step == 1
    for name in parameters:
        expected_first_moment = 0.1 * gradients[name]
        expected_second_moment = 0.001 * (gradients[name] ** 2)
        expected_updated = (
            parameters[name]
            - 0.1
            * gradients[name]
            / (np.sqrt(gradients[name] ** 2) + 1e-8)
        )

        assert np.allclose(optimizer.first_moment[name], expected_first_moment)
        assert np.allclose(optimizer.second_moment[name], expected_second_moment)
        assert np.allclose(updated_parameters[name], expected_updated)
        assert np.all(
            updated_parameters[name][gradients[name] > 0]
            < parameters[name][gradients[name] > 0]
        )
        assert np.all(
            updated_parameters[name][gradients[name] < 0]
            > parameters[name][gradients[name] < 0]
        )

    assert_dict_arrays_unchanged(parameters, original_parameters)


def test_parameter_adam_second_step_accumulates_state() -> None:
    parameters, gradients = create_parameter_case()
    optimizer = ParameterAdam(
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )

    optimizer.step(parameters, gradients)
    optimizer.step(parameters, gradients)

    assert optimizer.time_step == 2
    assert set(optimizer.first_moment) == set(parameters)
    assert set(optimizer.second_moment) == set(parameters)
    for name in parameters:
        expected_first_moment = 0.19 * gradients[name]
        expected_second_moment = 0.001999 * (gradients[name] ** 2)

        assert optimizer.first_moment[name].shape == parameters[name].shape
        assert optimizer.second_moment[name].shape == parameters[name].shape
        assert np.allclose(optimizer.first_moment[name], expected_first_moment)
        assert np.allclose(optimizer.second_moment[name], expected_second_moment)


def test_parameter_adam_rejects_incompatible_state() -> None:
    parameters, gradients = create_parameter_case()
    optimizer = ParameterAdam()
    optimizer.step(parameters, gradients)
    optimizer.first_moment["W1"] = np.zeros(3)

    with pytest.raises(ValueError):
        optimizer.step(parameters, gradients)


def test_dictionary_key_mismatch() -> None:
    parameters, gradients = create_parameter_case()
    missing_key_gradients = {
        "W1": gradients["W1"],
    }
    extra_key_gradients = {
        **gradients,
        "extra": np.array([1.0]),
    }
    optimizer = ParameterSGD()

    with pytest.raises(ValueError):
        optimizer.step(parameters, missing_key_gradients)

    with pytest.raises(ValueError):
        optimizer.step(parameters, extra_key_gradients)


def test_shape_mismatch() -> None:
    parameters, gradients = create_parameter_case()
    gradients["W1"] = np.array([1.0, 2.0])
    optimizer = ParameterSGD()

    with pytest.raises(ValueError):
        optimizer.step(parameters, gradients)


def test_invalid_non_array_values() -> None:
    parameters, gradients = create_parameter_case()
    optimizer = ParameterSGD()

    parameters["W1"] = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(TypeError):
        optimizer.step(parameters, gradients)

    parameters, gradients = create_parameter_case()
    gradients["b1"] = [0.2, -0.1]
    with pytest.raises(TypeError):
        optimizer.step(parameters, gradients)


def test_non_finite_values() -> None:
    parameters, gradients = create_parameter_case()
    parameters["W1"][0, 0] = np.nan
    optimizer = ParameterSGD()

    with pytest.raises(ValueError):
        optimizer.step(parameters, gradients)

    parameters, gradients = create_parameter_case()
    gradients["b1"][0] = np.inf
    with pytest.raises(ValueError):
        optimizer.step(parameters, gradients)


@pytest.mark.parametrize(
    ("factory", "expected_exception"),
    [
        (lambda: ParameterSGD(learning_rate=0.0), ValueError),
        (lambda: ParameterSGD(learning_rate=-0.1), ValueError),
        (lambda: ParameterSGD(learning_rate="0.1"), TypeError),
        (lambda: ParameterSGD(learning_rate=True), TypeError),
        (lambda: ParameterMomentum(learning_rate=0.0), ValueError),
        (lambda: ParameterMomentum(beta=-0.1), ValueError),
        (lambda: ParameterMomentum(beta=1.0), ValueError),
        (lambda: ParameterMomentum(beta=True), TypeError),
        (lambda: ParameterAdam(learning_rate=0.0), ValueError),
        (lambda: ParameterAdam(beta1=1.0), ValueError),
        (lambda: ParameterAdam(beta2=-0.1), ValueError),
        (lambda: ParameterAdam(epsilon=0.0), ValueError),
        (lambda: ParameterAdam(epsilon=False), TypeError),
    ],
)
def test_invalid_hyperparameters(factory, expected_exception) -> None:
    with pytest.raises(expected_exception):
        factory()
