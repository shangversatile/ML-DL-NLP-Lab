import numpy as np
import pytest

from src.evaluation.error_analysis import (
    misclassification_mask,
    select_high_confidence_errors,
    select_top_loss_examples,
    summarize_errors,
)


def _example_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 2, 2, 1])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.2, 0.7],
            [0.2, 0.6, 0.2],
        ]
    )
    return y_true, y_pred, probabilities


def test_misclassification_mask() -> None:
    y_true, y_pred, _ = _example_data()

    np.testing.assert_array_equal(
        misclassification_mask(y_true, y_pred),
        np.array([False, True, False, False]),
    )


def test_summarize_errors() -> None:
    y_true, y_pred, probabilities = _example_data()

    summary = summarize_errors(
        y_true,
        y_pred,
        probabilities,
        high_confidence_threshold=0.65,
    )

    assert summary["n_samples"] == 4
    assert summary["n_errors"] == 1
    assert summary["error_rate"] == pytest.approx(0.25)
    assert summary["n_high_confidence_errors"] == 1
    assert summary["mean_confidence_incorrect"] == pytest.approx(0.7)


def test_summarize_errors_returns_nan_for_empty_incorrect_set() -> None:
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])

    summary = summarize_errors(y_true, y_pred, probabilities)

    assert summary["n_errors"] == 0
    assert np.isnan(summary["mean_confidence_incorrect"])
    assert np.isnan(summary["mean_nll_incorrect"])
    assert np.isnan(summary["high_confidence_error_rate_among_errors"])


def test_select_top_loss_examples() -> None:
    y_true, y_pred, probabilities = _example_data()

    records = select_top_loss_examples(
        y_true,
        y_pred,
        probabilities,
        top_n=2,
    )

    expected_keys = {
        "index",
        "true_label",
        "predicted_label",
        "confidence",
        "true_class_probability",
        "negative_log_likelihood",
        "is_error",
    }

    assert len(records) == 2
    assert set(records[0]) == expected_keys
    assert records[0]["negative_log_likelihood"] >= records[1][
        "negative_log_likelihood"
    ]
    assert records[0]["index"] == 1
    assert records[0]["true_class_probability"] == pytest.approx(0.2)


def test_select_high_confidence_errors() -> None:
    y_true, y_pred, probabilities = _example_data()

    records = select_high_confidence_errors(
        y_true,
        y_pred,
        probabilities,
        threshold=0.65,
        top_n=5,
    )

    assert len(records) == 1
    assert records[0]["index"] == 1
    assert records[0]["confidence"] == pytest.approx(0.7)
    assert records[0]["is_error"] is True


def test_misclassification_mask_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        misclassification_mask([0, 1], np.array([0, 1]))

    with pytest.raises(ValueError):
        misclassification_mask(np.array([[0, 1]]), np.array([0, 1]))

    with pytest.raises(ValueError):
        misclassification_mask(np.array([0, 1]), np.array([0]))


@pytest.mark.parametrize(
    "threshold,expected_error",
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (True, TypeError),
        ("0.9", TypeError),
    ],
)
def test_invalid_thresholds(
    threshold: object,
    expected_error: type[Exception],
) -> None:
    y_true, y_pred, probabilities = _example_data()

    with pytest.raises(expected_error):
        summarize_errors(
            y_true,
            y_pred,
            probabilities,
            high_confidence_threshold=threshold,
        )

    with pytest.raises(expected_error):
        select_high_confidence_errors(
            y_true,
            y_pred,
            probabilities,
            threshold=threshold,
            top_n=1,
        )


@pytest.mark.parametrize(
    "top_n,expected_error",
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_invalid_top_n(
    top_n: object,
    expected_error: type[Exception],
) -> None:
    y_true, y_pred, probabilities = _example_data()

    with pytest.raises(expected_error):
        select_top_loss_examples(
            y_true,
            y_pred,
            probabilities,
            top_n=top_n,
        )

    with pytest.raises(expected_error):
        select_high_confidence_errors(
            y_true,
            y_pred,
            probabilities,
            threshold=0.65,
            top_n=top_n,
        )
