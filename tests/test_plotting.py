"""Tests for plotting utilities."""

import pytest

from src.utils.plotting import plot_loss_curve, plot_multiple_loss_curves


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
