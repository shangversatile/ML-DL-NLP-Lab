import numpy as np
import pytest

from experiments.analyze_mlp_overfitting import summarize_history
from src.data.preprocessing import flip_binary_labels


def test_summarize_history_finds_best_epoch() -> None:
    history = {
        "train_bce": [0.8, 0.5, 0.3],
        "val_bce": [0.7, 0.4, 0.6],
        "train_accuracy": [0.5, 0.7, 0.9],
        "val_accuracy": [0.5, 0.8, 0.75],
        "update_count": 12,
    }

    summary = summarize_history(history)

    assert summary["best_epoch"] == 2
    assert summary["best_val_bce"] == pytest.approx(0.4)
    assert summary["final_val_bce"] == pytest.approx(0.6)
    assert summary["post_best_val_bce_increase"] == pytest.approx(0.2)
    assert summary["final_train_bce"] == pytest.approx(0.3)
    assert summary["final_train_accuracy"] == pytest.approx(0.9)
    assert summary["final_val_accuracy"] == pytest.approx(0.75)


def test_summarize_history_handles_best_final_epoch() -> None:
    history = {
        "train_bce": [0.8, 0.5, 0.3],
        "val_bce": [0.7, 0.5, 0.4],
        "train_accuracy": [0.5, 0.7, 0.9],
        "val_accuracy": [0.5, 0.7, 0.8],
        "update_count": 12,
    }

    summary = summarize_history(history)

    assert summary["best_epoch"] == 3
    assert summary["best_val_bce"] == pytest.approx(0.4)
    assert summary["final_val_bce"] == pytest.approx(0.4)
    assert summary["post_best_val_bce_increase"] == pytest.approx(0.0)


def test_corruption_changes_expected_count() -> None:
    y = np.array([0, 1] * 25)

    corrupted_y = flip_binary_labels(y, flip_rate=0.3, seed=42)

    assert np.sum(y != corrupted_y) == 15
