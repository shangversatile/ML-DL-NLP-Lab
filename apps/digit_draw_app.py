"""Local Tkinter handwritten-digit drawing app for the scratch MLP."""

import argparse
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.digit_canvas_preprocessing import preprocess_canvas_image
from src.inference.digits_inference import predict_single_digit
from src.models.checkpoint import load_multiclass_mlp_checkpoint


DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/digits_mlp_baseline.npz")
CANVAS_SIZE = 280
BRUSH_RADIUS = 10


class DigitDrawApp:
    def __init__(
        self,
        root: tk.Tk,
        checkpoint_path: Path,
    ) -> None:
        self.root = root
        self.root.title("Scratch MLP Digit Draw App")
        self.model, self.metadata = load_multiclass_mlp_checkpoint(checkpoint_path)
        self.class_names = self.metadata.get(
            "class_names",
            [str(index) for index in range(self.model.num_classes)],
        )

        self.canvas_size = CANVAS_SIZE
        self.brush_radius = BRUSH_RADIUS
        self.last_position: tuple[int, int] | None = None
        self.canvas_buffer = np.full(
            (self.canvas_size, self.canvas_size),
            255.0,
            dtype=float,
        )

        self.result_text = tk.StringVar(
            value="Draw a digit, then click Predict.",
        )

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, columnspan=2, padx=12, pady=12)
        self.canvas.bind("<ButtonPress-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._end_draw)

        clear_button = tk.Button(root, text="Clear", command=self.clear)
        clear_button.grid(row=1, column=0, padx=8, pady=8, sticky="ew")

        predict_button = tk.Button(root, text="Predict", command=self.predict)
        predict_button.grid(row=1, column=1, padx=8, pady=8, sticky="ew")

        result_label = tk.Label(
            root,
            textvariable=self.result_text,
            justify="left",
            anchor="w",
            width=44,
        )
        result_label.grid(row=2, column=0, columnspan=2, padx=12, pady=12, sticky="w")

    def _start_draw(
        self,
        event: tk.Event,
    ) -> None:
        self.last_position = (int(event.x), int(event.y))
        self._stamp_circle(int(event.x), int(event.y))

    def _draw(
        self,
        event: tk.Event,
    ) -> None:
        current_position = (int(event.x), int(event.y))
        if self.last_position is None:
            self.last_position = current_position
            return

        x0, y0 = self.last_position
        x1, y1 = current_position
        self.canvas.create_line(
            x0,
            y0,
            x1,
            y1,
            fill="black",
            width=self.brush_radius * 2,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self._stamp_line(x0, y0, x1, y1)
        self.last_position = current_position

    def _end_draw(
        self,
        _event: tk.Event,
    ) -> None:
        self.last_position = None

    def _stamp_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> None:
        distance = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(distance + 1):
            fraction = step / distance
            x = int(round(x0 + fraction * (x1 - x0)))
            y = int(round(y0 + fraction * (y1 - y0)))
            self._stamp_circle(x, y)

    def _stamp_circle(
        self,
        x: int,
        y: int,
    ) -> None:
        radius = self.brush_radius
        row_min = max(0, y - radius)
        row_max = min(self.canvas_size - 1, y + radius)
        col_min = max(0, x - radius)
        col_max = min(self.canvas_size - 1, x + radius)

        rows = np.arange(row_min, row_max + 1)
        cols = np.arange(col_min, col_max + 1)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
        mask = (row_grid - y) ** 2 + (col_grid - x) ** 2 <= radius**2
        self.canvas_buffer[row_min : row_max + 1, col_min : col_max + 1][mask] = 0.0

    def clear(self) -> None:
        self.canvas.delete("all")
        self.canvas_buffer.fill(255.0)
        self.result_text.set("Draw a digit, then click Predict.")

    def predict(self) -> None:
        is_blank = bool(np.all(self.canvas_buffer == 255.0))
        feature_vector = preprocess_canvas_image(self.canvas_buffer)
        result = predict_single_digit(
            self.model,
            feature_vector,
            top_k=3,
        )

        predicted_index = int(result["prediction"])
        predicted_digit = self.class_names[predicted_index]
        lines = []
        if is_blank:
            lines.append("No clear digit detected; prediction may be meaningless.")

        lines.append(
            f"Prediction: {predicted_digit} "
            f"(confidence {result['confidence']:.3f})"
        )
        lines.append("Top-3 candidates:")

        for class_index, probability in zip(
            result["top_k_indices"],
            result["top_k_probabilities"],
        ):
            digit_name = self.class_names[int(class_index)]
            lines.append(f"  {digit_name}: {float(probability):.3f}")

        self.result_text.set("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a digit and classify it with the scratch MLP checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "Path to a saved MulticlassMLPScratch checkpoint "
            f"(default: {DEFAULT_CHECKPOINT_PATH})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint

    if not checkpoint_path.exists():
        print(
            "Checkpoint not found: "
            f"{checkpoint_path}\n"
            "Run this first:\n"
            "  python experiments/train_save_load_digits_mlp.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    root = tk.Tk()
    DigitDrawApp(root, checkpoint_path)
    root.mainloop()


if __name__ == "__main__":
    main()
