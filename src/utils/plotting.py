"""Plotting helpers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _create_parent_directory(
    output_path: str,
) -> Path:
    path = Path(output_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


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

    path = _create_parent_directory(output_path)

    epochs = np.arange(1, losses.size + 1)

    plt.figure()
    plt.plot(epochs, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()


def plot_multiple_loss_curves(
    histories: dict[str, list[float] | np.ndarray],
    output_path: str,
    title: str = "Optimizer Comparison",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> None:
    """
    Plot and save multiple loss curves in one figure.
    """
    if not isinstance(histories, dict) or len(histories) == 0:
        raise ValueError("histories must be a non-empty dictionary.")

    converted_histories = {}
    for name, history in histories.items():
        losses = np.asarray(history, dtype=float)

        if losses.size == 0:
            raise ValueError("each history must not be empty.")
        if not np.all(np.isfinite(losses)):
            raise ValueError("histories must contain only finite values.")

        converted_histories[name] = losses.reshape(-1)

    path = _create_parent_directory(output_path)

    plt.figure()
    for name, losses in converted_histories.items():
        epochs = np.arange(1, losses.size + 1)
        plt.plot(epochs, losses, label=name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: str,
    class_names: list[str] | None = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a NumPy array.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array.")
    if matrix.shape[0] == 0:
        raise ValueError("matrix must not be empty.")
    if np.issubdtype(matrix.dtype, np.bool_) or not np.issubdtype(
        matrix.dtype,
        np.number,
    ):
        raise ValueError("matrix must contain numeric values.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must contain only finite values.")
    if np.any(matrix < 0):
        raise ValueError("matrix values must be non-negative.")

    if class_names is None:
        class_names = [str(index) for index in range(matrix.shape[0])]
    if len(class_names) != matrix.shape[0]:
        raise ValueError("class_names length must match matrix size.")

    display_matrix = matrix.astype(float)
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.zeros_like(display_matrix, dtype=float)
        np.divide(
            display_matrix,
            row_sums,
            out=normalized_matrix,
            where=row_sums > 0,
        )
        display_matrix = normalized_matrix

    path = _create_parent_directory(output_path)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(display_matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    tick_positions = np.arange(matrix.shape[0])
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(class_names)
    axis.set_yticklabels(class_names)

    max_value = display_matrix.max() if display_matrix.size > 0 else 0.0
    threshold = max_value / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = display_matrix[row, column]
            annotation = f"{value:.2f}" if normalize else f"{int(matrix[row, column])}"
            text_color = "white" if value > threshold else "black"
            axis.text(
                column,
                row,
                annotation,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_digit_examples(
    images: np.ndarray,
    titles: list[str],
    output_path: str,
    n_cols: int = 5,
    figure_title: str = "Digit Examples",
) -> None:
    if not isinstance(images, np.ndarray):
        raise TypeError("images must be a NumPy array.")
    if images.ndim != 3 or images.shape[1:] != (8, 8):
        raise ValueError("images must have shape (n_examples, 8, 8).")
    if len(titles) != images.shape[0]:
        raise ValueError("titles length must match number of images.")
    if type(n_cols) is not int:
        raise TypeError("n_cols must be an integer.")
    if n_cols <= 0:
        raise ValueError("n_cols must be positive.")

    path = _create_parent_directory(output_path)
    n_examples = images.shape[0]

    if n_examples == 0:
        figure, axis = plt.subplots(figsize=(5, 2.5))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No examples available",
            ha="center",
            va="center",
            fontsize=12,
        )
        figure.suptitle(figure_title)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
        return

    n_cols = min(n_cols, n_examples)
    n_rows = int(np.ceil(n_examples / n_cols))
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.8 * n_rows),
    )
    axes_array = np.asarray(axes).reshape(-1)

    for index, axis in enumerate(axes_array):
        axis.axis("off")
        if index >= n_examples:
            continue

        axis.imshow(images[index], cmap="gray")
        axis.set_title(titles[index], fontsize=8)

    figure.suptitle(figure_title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
