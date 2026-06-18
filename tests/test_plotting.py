"""Tests for plotting utilities."""

import numpy as np
import pytest

from src.utils.plotting import (
    plot_confidence_bin_summary,
    plot_confusion_matrix,
    plot_digit_examples,
    plot_grouped_metric_bars,
    plot_loss_curve,
    plot_multiple_loss_curves,
    plot_shift_metric_bars,
)


def test_plot_loss_curve_creates_file(tmp_path) -> None:
    loss_history = [3.0, 2.0, 1.0]
    output_path = tmp_path / "loss_curve.png"

    plot_loss_curve(loss_history, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_loss_curve_empty_history(tmp_path) -> None:
    output_path = tmp_path / "loss_curve.png"

    with pytest.raises(ValueError):
        plot_loss_curve([], str(output_path))


def test_plot_loss_curve_non_finite_loss(tmp_path) -> None:
    output_path = tmp_path / "loss_curve.png"

    with pytest.raises(ValueError):
        plot_loss_curve([1.0, float("nan")], str(output_path))

    with pytest.raises(ValueError):
        plot_loss_curve([1.0, float("inf")], str(output_path))


def test_plot_multiple_loss_curves_creates_file(tmp_path) -> None:
    histories = {
        "sgd": [1.0, 0.8, 0.6],
        "adam": [1.0, 0.5, 0.3],
    }
    output_path = tmp_path / "comparison.png"

    plot_multiple_loss_curves(histories, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_multiple_loss_curves_empty_histories(tmp_path) -> None:
    output_path = tmp_path / "comparison.png"

    with pytest.raises(ValueError):
        plot_multiple_loss_curves({}, str(output_path))


def test_plot_multiple_loss_curves_empty_individual_history(tmp_path) -> None:
    output_path = tmp_path / "comparison.png"

    with pytest.raises(ValueError):
        plot_multiple_loss_curves({"sgd": []}, str(output_path))


def test_plot_multiple_loss_curves_non_finite_values(tmp_path) -> None:
    output_path = tmp_path / "comparison.png"

    with pytest.raises(ValueError):
        plot_multiple_loss_curves({"sgd": [1.0, float("nan")]}, str(output_path))

    with pytest.raises(ValueError):
        plot_multiple_loss_curves({"adam": [1.0, float("inf")]}, str(output_path))


def test_plot_confusion_matrix_creates_file_for_counts(tmp_path) -> None:
    matrix = np.array([[2, 1], [0, 3]])
    output_path = tmp_path / "confusion_counts.png"

    plot_confusion_matrix(matrix, str(output_path), class_names=["0", "1"])

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_confusion_matrix_creates_file_for_normalized_matrix(tmp_path) -> None:
    matrix = np.array([[2, 1], [0, 3]])
    output_path = tmp_path / "confusion_normalized.png"

    plot_confusion_matrix(
        matrix,
        str(output_path),
        class_names=["0", "1"],
        normalize=True,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_confusion_matrix_invalid_matrix_raises_errors(tmp_path) -> None:
    output_path = tmp_path / "confusion.png"

    with pytest.raises(TypeError):
        plot_confusion_matrix([[1, 0], [0, 1]], str(output_path))

    with pytest.raises(ValueError):
        plot_confusion_matrix(np.array([[1, 0, 0], [0, 1, 0]]), str(output_path))

    with pytest.raises(ValueError):
        plot_confusion_matrix(np.array([[1, -1], [0, 1]]), str(output_path))

    with pytest.raises(ValueError):
        plot_confusion_matrix(np.array([[1, np.nan], [0, 1]]), str(output_path))

    with pytest.raises(ValueError):
        plot_confusion_matrix(
            np.array([[1, 0], [0, 1]]),
            str(output_path),
            class_names=["0"],
        )


def test_plot_digit_examples_creates_file_for_non_empty_examples(tmp_path) -> None:
    images = np.zeros((3, 8, 8))
    titles = ["first", "second", "third"]
    output_path = tmp_path / "digits.png"

    plot_digit_examples(images, titles, str(output_path), n_cols=2)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_digit_examples_creates_file_for_zero_examples(tmp_path) -> None:
    images = np.zeros((0, 8, 8))
    output_path = tmp_path / "empty_digits.png"

    plot_digit_examples(images, [], str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_digit_examples_mismatched_titles_raise_value_error(tmp_path) -> None:
    images = np.zeros((2, 8, 8))
    output_path = tmp_path / "digits.png"

    with pytest.raises(ValueError):
        plot_digit_examples(images, ["only one title"], str(output_path))


def test_plot_digit_examples_invalid_image_shape_raises_value_error(tmp_path) -> None:
    output_path = tmp_path / "digits.png"

    with pytest.raises(ValueError):
        plot_digit_examples(np.zeros((2, 7, 8)), ["a", "b"], str(output_path))

    with pytest.raises(ValueError):
        plot_digit_examples(np.zeros((8, 8)), ["a"], str(output_path))


def test_plot_shift_metric_bars_creates_file(tmp_path) -> None:
    output_path = tmp_path / "shift_bars.png"

    plot_shift_metric_bars(
        ["clean", "shifted"],
        [0.95, 0.80],
        str(output_path),
        ylabel="Accuracy",
        title="Shift Accuracy",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_shift_metric_bars_invalid_inputs(tmp_path) -> None:
    output_path = tmp_path / "shift_bars.png"

    with pytest.raises(ValueError):
        plot_shift_metric_bars(
            ["clean"],
            [0.95, 0.80],
            str(output_path),
            ylabel="Accuracy",
            title="Shift Accuracy",
        )

    with pytest.raises(ValueError):
        plot_shift_metric_bars(
            [],
            [],
            str(output_path),
            ylabel="Accuracy",
            title="Shift Accuracy",
        )

    with pytest.raises(ValueError):
        plot_shift_metric_bars(
            ["clean"],
            [float("nan")],
            str(output_path),
            ylabel="Accuracy",
            title="Shift Accuracy",
        )


def test_plot_confidence_bin_summary_creates_file(tmp_path) -> None:
    output_path = tmp_path / "confidence_bins.png"
    bin_summary = [
        {
            "bin_index": 0,
            "lower": 0.0,
            "upper": 0.5,
            "count": 0,
            "accuracy": np.nan,
            "mean_confidence": np.nan,
            "confidence_gap": np.nan,
        },
        {
            "bin_index": 1,
            "lower": 0.5,
            "upper": 1.0,
            "count": 3,
            "accuracy": 2 / 3,
            "mean_confidence": 0.8,
            "confidence_gap": 0.8 - 2 / 3,
        },
    ]

    plot_confidence_bin_summary(bin_summary, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_confidence_bin_summary_handles_empty_bins(tmp_path) -> None:
    output_path = tmp_path / "confidence_bins_empty.png"
    bin_summary = [
        {
            "bin_index": 0,
            "lower": 0.0,
            "upper": 0.5,
            "count": 0,
            "accuracy": np.nan,
            "mean_confidence": np.nan,
            "confidence_gap": np.nan,
        },
    ]

    plot_confidence_bin_summary(bin_summary, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_confidence_bin_summary_rejects_empty_summary(tmp_path) -> None:
    output_path = tmp_path / "confidence_bins.png"

    with pytest.raises(ValueError):
        plot_confidence_bin_summary([], str(output_path))


def test_plot_grouped_metric_bars_creates_file(tmp_path) -> None:
    output_path = tmp_path / "grouped_bars.png"

    plot_grouped_metric_bars(
        ["clean", "thicken"],
        {
            "Baseline": [0.98, 0.18],
            "Augmented": [0.96, 0.80],
        },
        str(output_path),
        ylabel="Accuracy",
        title="Baseline vs Augmented",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_grouped_metric_bars_rejects_invalid_inputs(tmp_path) -> None:
    output_path = tmp_path / "grouped_bars.png"

    with pytest.raises(ValueError):
        plot_grouped_metric_bars(
            [],
            {"Baseline": []},
            str(output_path),
            ylabel="Accuracy",
            title="Invalid",
        )

    with pytest.raises(ValueError):
        plot_grouped_metric_bars(
            ["clean"],
            {},
            str(output_path),
            ylabel="Accuracy",
            title="Invalid",
        )

    with pytest.raises(ValueError):
        plot_grouped_metric_bars(
            ["clean", "thicken"],
            {"Baseline": [0.98]},
            str(output_path),
            ylabel="Accuracy",
            title="Invalid",
        )

    with pytest.raises(ValueError):
        plot_grouped_metric_bars(
            ["clean"],
            {"Baseline": [float("nan")]},
            str(output_path),
            ylabel="Accuracy",
            title="Invalid",
        )
