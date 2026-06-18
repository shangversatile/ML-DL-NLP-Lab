import numpy as np
import pytest

from src.evaluation.confidence_diagnostics import (
    confidence_bin_summary,
    expected_calibration_error,
    summarize_confidence_behavior,
)


def _example_data() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0, 1, 1, 0])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
        ]
    )
    return y_true, probabilities


def test_confidence_bin_summary() -> None:
    y_true, probabilities = _example_data()

    summary = confidence_bin_summary(y_true, probabilities, n_bins=5)

    assert sum(record["count"] for record in summary) == 4
    non_empty_bins = [record for record in summary if record["count"] > 0]
    assert non_empty_bins
    for record in non_empty_bins:
        assert np.isfinite(record["accuracy"])
        assert np.isfinite(record["mean_confidence"])
        assert record["confidence_gap"] == pytest.approx(
            record["mean_confidence"] - record["accuracy"]
        )


def test_expected_calibration_error() -> None:
    y_true, probabilities = _example_data()
    summary = confidence_bin_summary(y_true, probabilities, n_bins=5)

    ece = expected_calibration_error(summary, n_samples=4)

    assert np.isfinite(ece)
    assert ece >= 0.0


def test_summarize_confidence_behavior() -> None:
    y_true, probabilities = _example_data()

    summary = summarize_confidence_behavior(y_true, probabilities, n_bins=5)

    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["mean_confidence"] == pytest.approx(0.75)
    assert summary["overconfidence_gap"] == pytest.approx(0.25)
    assert summary["mean_confidence_correct"] == pytest.approx(0.8)
    assert summary["mean_confidence_incorrect"] == pytest.approx(0.7)
    assert summary["ece"] >= 0.0


def test_empty_bins_are_nan() -> None:
    y_true, probabilities = _example_data()

    summary = confidence_bin_summary(y_true, probabilities, n_bins=20)

    empty_bins = [record for record in summary if record["count"] == 0]
    assert empty_bins
    assert all(np.isnan(record["accuracy"]) for record in empty_bins)
    assert all(np.isnan(record["mean_confidence"]) for record in empty_bins)
    assert all(np.isnan(record["confidence_gap"]) for record in empty_bins)


def test_invalid_probability_matrix() -> None:
    y_true, _ = _example_data()

    with pytest.raises(TypeError):
        confidence_bin_summary(y_true, [[0.5, 0.5]], n_bins=5)

    with pytest.raises(ValueError):
        confidence_bin_summary(y_true, np.array([0.5, 0.5]), n_bins=5)

    with pytest.raises(ValueError):
        confidence_bin_summary(
            y_true,
            np.array([[0.5, 0.5], [0.3, np.nan], [0.5, 0.5], [0.5, 0.5]]),
            n_bins=5,
        )

    with pytest.raises(ValueError):
        confidence_bin_summary(
            y_true,
            np.array([[0.5, 0.4], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]]),
            n_bins=5,
        )


def test_invalid_labels() -> None:
    _, probabilities = _example_data()

    with pytest.raises(TypeError):
        confidence_bin_summary([0, 1, 1, 0], probabilities, n_bins=5)

    with pytest.raises(ValueError):
        confidence_bin_summary(np.array([[0, 1, 1, 0]]), probabilities, n_bins=5)

    with pytest.raises(ValueError):
        confidence_bin_summary(np.array([0.0, 1.0, 1.0, 0.0]), probabilities, n_bins=5)

    with pytest.raises(ValueError):
        confidence_bin_summary(np.array([0, 1, 2, 0]), probabilities, n_bins=5)


@pytest.mark.parametrize(
    "n_bins,expected_error",
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (2.5, TypeError),
    ],
)
def test_invalid_n_bins(
    n_bins: object,
    expected_error: type[Exception],
) -> None:
    y_true, probabilities = _example_data()

    with pytest.raises(expected_error):
        confidence_bin_summary(y_true, probabilities, n_bins=n_bins)


@pytest.mark.parametrize(
    "n_samples,expected_error",
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (2.5, TypeError),
    ],
)
def test_invalid_n_samples_for_ece(
    n_samples: object,
    expected_error: type[Exception],
) -> None:
    y_true, probabilities = _example_data()
    summary = confidence_bin_summary(y_true, probabilities, n_bins=5)

    with pytest.raises(expected_error):
        expected_calibration_error(summary, n_samples=n_samples)


def test_empty_bin_summary_for_ece() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([], n_samples=4)
