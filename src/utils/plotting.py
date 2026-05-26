"""Plotting helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(
    loss_history: list[float] | np.ndarray,
    output_path: str,
    title: str = "Training Loss Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> None:
    """Plot and save a training loss curve."""
    losses = np.asarray(loss_history, dtype=float)

    if losses.size == 0:
        raise ValueError("loss_history must not be empty.")
    if not np.all(np.isfinite(losses)):
        raise ValueError("loss_history must contain only finite values.")

    path = Path(output_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, losses.size + 1)

    plt.figure()
    plt.plot(epochs, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()
