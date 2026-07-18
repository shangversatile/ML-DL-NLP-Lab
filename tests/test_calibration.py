import numpy as np
import pytest

from src.evaluation.calibration import (
    calibration_bin_summary,
    calibration_summary,
    expected_calibration_error,
    maximum_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood,
    prediction_confidence_and_correctness,
    validate_probabilities_and_labels,
)


def _example_probabilities_and_labels() -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    y_true = np.array([0, 1, 1, 0])
    return probabilities, y_true


def test_validate_probabilities_and_labels_valid_inputs_pass() -> None:
    probabilities = np.array([[0.7, 0.3], [0.2, 0.8]])
    y_true = np.array([0.0, 1.0])

    validated_probabilities, validated_labels = validate_probabilities_and_labels(
        probabilities,
        y_true,
    )

    np.testing.assert_allclose(validated_probabilities, probabilities)
    np.testing.assert_array_equal(validated_labels, np.array([0, 1]))
    assert not np.shares_memory(validated_probabilities, probabilities)
    assert not np.shares_memory(validated_labels, y_true)


def test_validate_probabilities_and_labels_mismatched_sample_counts_fail() -> None:
    with pytest.raises(ValueError, match="matching sample counts"):
        validate_probabilities_and_labels(
            np.array([[0.7, 0.3], [0.2, 0.8]]),
            np.array([0]),
        )


def test_validate_probabilities_and_labels_non_2d_probabilities_fail() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_probabilities_and_labels(
            np.array([0.7, 0.3]),
            np.array([0]),
        )


def test_validate_probabilities_and_labels_non_1d_labels_fail() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_probabilities_and_labels(
            np.array([[0.7, 0.3]]),
            np.array([[0]]),
        )


def test_validate_probabilities_and_labels_empty_arrays_fail() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        validate_probabilities_and_labels(
            np.empty((0, 2)),
            np.array([], dtype=int),
        )

    with pytest.raises(ValueError, match="must not be empty"):
        validate_probabilities_and_labels(
            np.array([[0.7, 0.3]]),
            np.array([], dtype=int),
        )


def test_validate_probabilities_and_labels_non_finite_probabilities_fail() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_probabilities_and_labels(
            np.array([[0.7, np.nan], [0.2, 0.8]]),
            np.array([0, 1]),
        )


def test_validate_probabilities_and_labels_labels_out_of_range_fail() -> None:
    with pytest.raises(ValueError, match="0 <= label"):
        validate_probabilities_and_labels(
            np.array([[0.7, 0.3], [0.2, 0.8]]),
            np.array([0, 2]),
        )


def test_validate_probabilities_and_labels_rows_not_summing_to_one_fail() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        validate_probabilities_and_labels(
            np.array([[0.7, 0.2], [0.2, 0.8]]),
            np.array([0, 1]),
        )


def test_validate_probabilities_and_labels_non_integer_labels_fail() -> None:
    with pytest.raises(ValueError, match="integer-like"):
        validate_probabilities_and_labels(
            np.array([[0.7, 0.3], [0.2, 0.8]]),
            np.array([0.5, 1.0]),
        )


def test_calibration_bin_summary_rejects_bool_n_bins() -> None:
    probabilities, y_true = _example_probabilities_and_labels()

    with pytest.raises(TypeError, match="n_bins"):
        calibration_bin_summary(probabilities, y_true, n_bins=True)


def test_prediction_confidence_and_correctness() -> None:
    probabilities, y_true = _example_probabilities_and_labels()

    predictions, confidences, correctness = prediction_confidence_and_correctness(
        probabilities,
        y_true,
    )

    np.testing.assert_array_equal(predictions, np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(confidences, np.array([0.8, 0.6, 0.7, 0.9]))
    assert correctness.dtype == bool
    np.testing.assert_array_equal(correctness, np.array([True, False, True, False]))


def test_calibration_bin_summary_counts_bins_and_signed_gap() -> None:
    probabilities, y_true = _example_probabilities_and_labels()

    summary = calibration_bin_summary(probabilities, y_true, n_bins=5)

    assert len(summary) == 5
    assert summary[3]["count"] == 2
    assert summary[3]["accuracy"] == pytest.approx(0.5)
    assert summary[3]["confidence"] == pytest.approx(0.65)
    assert summary[3]["gap"] == pytest.approx(0.15)
    assert summary[4]["count"] == 2
    assert summary[4]["accuracy"] == pytest.approx(0.5)
    assert summary[4]["confidence"] == pytest.approx(0.85)
    assert summary[4]["gap"] == pytest.approx(0.35)


def test_calibration_bin_summary_includes_confidence_one_in_last_bin() -> None:
    probabilities = np.array([[1.0, 0.0], [0.6, 0.4]])
    y_true = np.array([0, 0])

    summary = calibration_bin_summary(probabilities, y_true, n_bins=10)

    assert summary[-1]["count"] == 1
    assert summary[-1]["confidence"] == pytest.approx(1.0)


def test_calibration_bin_summary_empty_bins_are_nan() -> None:
    probabilities = np.array([[1.0, 0.0], [0.6, 0.4]])
    y_true = np.array([0, 0])

    summary = calibration_bin_summary(probabilities, y_true, n_bins=10)
    empty_bins = [record for record in summary if record["count"] == 0]

    assert empty_bins
    assert all(np.isnan(record["accuracy"]) for record in empty_bins)
    assert all(np.isnan(record["confidence"]) for record in empty_bins)
    assert all(np.isnan(record["gap"]) for record in empty_bins)


def test_ece_and_mce_use_manually_computed_bin_gaps() -> None:
    probabilities, y_true = _example_probabilities_and_labels()

    assert expected_calibration_error(probabilities, y_true, n_bins=5) == (
        pytest.approx(0.25)
    )
    assert maximum_calibration_error(probabilities, y_true, n_bins=5) == (
        pytest.approx(0.35)
    )


def test_perfect_predictions_can_have_zero_ece() -> None:
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 1])

    assert expected_calibration_error(probabilities, y_true, n_bins=10) == (
        pytest.approx(0.0)
    )
    assert maximum_calibration_error(probabilities, y_true, n_bins=10) == (
        pytest.approx(0.0)
    )


def test_high_confidence_wrong_predictions_have_high_ece() -> None:
    probabilities = np.array([[0.99, 0.01], [0.98, 0.02]])
    y_true = np.array([1, 1])

    assert expected_calibration_error(probabilities, y_true, n_bins=10) == (
        pytest.approx(0.985)
    )


def test_multiclass_brier_score_manual_value() -> None:
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])
    y_true = np.array([0, 1])

    assert multiclass_brier_score(probabilities, y_true) == pytest.approx(0.2)


def test_multiclass_brier_score_perfect_prediction_is_zero() -> None:
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 1])

    assert multiclass_brier_score(probabilities, y_true) == pytest.approx(0.0)


def test_negative_log_likelihood_manual_value() -> None:
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])
    y_true = np.array([0, 1])

    expected = float(np.mean([-np.log(0.8), -np.log(0.6)]))
    assert negative_log_likelihood(probabilities, y_true) == pytest.approx(expected)


def test_negative_log_likelihood_perfect_high_probability_is_small() -> None:
    probabilities = np.array([[0.999, 0.001], [0.001, 0.999]])
    y_true = np.array([0, 1])

    assert negative_log_likelihood(probabilities, y_true) < 0.002


def test_negative_log_likelihood_wrong_high_confidence_is_large() -> None:
    probabilities = np.array([[0.999, 0.001], [0.998, 0.002]])
    y_true = np.array([1, 1])

    assert negative_log_likelihood(probabilities, y_true) > 3.0


def test_calibration_summary_includes_expected_keys_and_gap() -> None:
    probabilities, y_true = _example_probabilities_and_labels()

    summary = calibration_summary(probabilities, y_true, n_bins=5)

    assert set(summary) == {
        "n_samples",
        "accuracy",
        "mean_confidence",
        "mean_confidence_correct",
        "mean_confidence_incorrect",
        "overconfidence_gap",
        "ece",
        "mce",
        "brier_score",
        "nll",
    }
    assert summary["n_samples"] == 4
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["mean_confidence"] == pytest.approx(0.75)
    assert summary["overconfidence_gap"] == pytest.approx(0.25)
    assert summary["ece"] == pytest.approx(0.25)


def test_calibration_summary_handles_all_correct_edge_case() -> None:
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 1])

    summary = calibration_summary(probabilities, y_true, n_bins=10)

    assert summary["accuracy"] == pytest.approx(1.0)
    assert summary["mean_confidence_correct"] == pytest.approx(1.0)
    assert np.isnan(summary["mean_confidence_incorrect"])


def test_calibration_summary_handles_all_incorrect_edge_case() -> None:
    probabilities = np.array([[1.0, 0.0], [1.0, 0.0]])
    y_true = np.array([1, 1])

    summary = calibration_summary(probabilities, y_true, n_bins=10)

    assert summary["accuracy"] == pytest.approx(0.0)
    assert np.isnan(summary["mean_confidence_correct"])
    assert summary["mean_confidence_incorrect"] == pytest.approx(1.0)
