"""Local Tkinter handwritten-digit drawing app for the scratch MLP."""

import argparse
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.canvas_sample_store import save_canvas_sample
from src.inference.digit_canvas_preprocessing import (
    preprocess_canvas_image_with_stages,
)
from src.inference.digits_inference import predict_single_digit
from src.models.checkpoint import load_multiclass_mlp_checkpoint


AUGMENTED_CHECKPOINT_PATH = Path("results/checkpoints/digits_mlp_augmented.npz")
BASELINE_CHECKPOINT_PATH = Path("results/checkpoints/digits_mlp_baseline.npz")
DEFAULT_SAMPLE_DIR = Path("data/user_digits/samples")
CANVAS_SIZE = 280
BRUSH_RADIUS = 10
DEBUG_CELL_SIZE = 20


def _project_path(
    path: Path,
) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_checkpoint_path(
    cli_checkpoint: Path | None,
) -> tuple[Path | None, str, str | None]:
    if cli_checkpoint is not None:
        checkpoint_path = _project_path(cli_checkpoint)
        return checkpoint_path, "custom", None

    augmented_path = _project_path(AUGMENTED_CHECKPOINT_PATH)
    if augmented_path.exists():
        return augmented_path, "augmented", None

    baseline_path = _project_path(BASELINE_CHECKPOINT_PATH)
    if baseline_path.exists():
        return (
            baseline_path,
            "baseline",
            "Using baseline checkpoint because augmented checkpoint was not found.",
        )

    return None, "missing", None


class DigitDrawApp:
    def __init__(
        self,
        root: tk.Tk,
        checkpoint_path: Path,
        checkpoint_type: str,
        checkpoint_warning: str | None = None,
    ) -> None:
        self.root = root
        self.root.title("Scratch MLP Digit Draw App")
        self.checkpoint_path = checkpoint_path
        self.checkpoint_type = checkpoint_type
        self.model, self.metadata = load_multiclass_mlp_checkpoint(checkpoint_path)
        self.class_names = self.metadata.get(
            "class_names",
            [str(index) for index in range(self.model.num_classes)],
        )

        self.canvas_size = CANVAS_SIZE
        self.brush_radius = BRUSH_RADIUS
        self.last_position: tuple[int, int] | None = None
        self.last_preprocessing_stages = None
        self.last_prediction_result = None
        self.canvas_buffer = np.full(
            (self.canvas_size, self.canvas_size),
            255.0,
            dtype=float,
        )

        self.result_text = tk.StringVar(
            value="Draw a digit, then click Predict.",
        )
        self.debug_text = tk.StringVar(
            value="Model input after preprocessing (8x8)\nNo prediction yet.",
        )
        self.save_status_text = tk.StringVar(value="")
        self.checkpoint_text = tk.StringVar(
            value=(
                f"Checkpoint type: {checkpoint_type}\n"
                f"Loaded checkpoint: {checkpoint_path}"
            )
        )
        if checkpoint_warning is not None:
            self.checkpoint_text.set(
                self.checkpoint_text.get() + f"\nWarning: {checkpoint_warning}"
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

        debug_frame = tk.Frame(root)
        debug_frame.grid(row=0, column=2, padx=12, pady=12, sticky="n")
        tk.Label(
            debug_frame,
            text="Model input after preprocessing (8x8)",
            justify="left",
        ).pack(anchor="w")
        debug_size = DEBUG_CELL_SIZE * 8
        self.debug_canvas = tk.Canvas(
            debug_frame,
            width=debug_size,
            height=debug_size,
            bg="black",
        )
        self.debug_canvas.pack(pady=6)
        tk.Label(
            debug_frame,
            textvariable=self.debug_text,
            justify="left",
            anchor="w",
            width=38,
        ).pack(anchor="w")

        clear_button = tk.Button(root, text="Clear", command=self.clear)
        clear_button.grid(row=1, column=0, padx=8, pady=8, sticky="ew")

        predict_button = tk.Button(root, text="Predict", command=self.predict)
        predict_button.grid(row=1, column=1, padx=8, pady=8, sticky="ew")

        save_frame = tk.Frame(root)
        save_frame.grid(row=1, column=2, padx=8, pady=8, sticky="ew")
        tk.Label(save_frame, text="True label optional").grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.true_label_entry = tk.Entry(save_frame, width=6)
        self.true_label_entry.grid(row=0, column=1, padx=6)
        save_button = tk.Button(save_frame, text="Save sample", command=self.save_sample)
        save_button.grid(row=0, column=2, padx=6)

        result_label = tk.Label(
            root,
            textvariable=self.result_text,
            justify="left",
            anchor="w",
            width=54,
        )
        result_label.grid(row=2, column=0, columnspan=2, padx=12, pady=12, sticky="w")

        checkpoint_label = tk.Label(
            root,
            textvariable=self.checkpoint_text,
            justify="left",
            anchor="w",
            width=72,
        )
        checkpoint_label.grid(row=2, column=2, padx=12, pady=12, sticky="w")

        guidance = tk.Label(
            root,
            text=(
                "Debug rule:\n"
                "If the 8x8 model input does not resemble your digit, "
                "the issue is preprocessing.\n"
                "If the 8x8 input looks correct but prediction is wrong, "
                "the issue is model/data coverage."
            ),
            justify="left",
            anchor="w",
        )
        guidance.grid(row=3, column=0, columnspan=3, padx=12, pady=8, sticky="w")

        save_status_label = tk.Label(
            root,
            textvariable=self.save_status_text,
            justify="left",
            anchor="w",
            width=100,
        )
        save_status_label.grid(row=4, column=0, columnspan=3, padx=12, pady=8, sticky="w")

        self._draw_debug_grid(np.zeros((8, 8), dtype=float))

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

    def _draw_debug_grid(
        self,
        image_8x8: np.ndarray,
    ) -> None:
        self.debug_canvas.delete("all")
        clipped = np.clip(np.asarray(image_8x8, dtype=float), 0.0, 1.0)
        for row in range(8):
            for col in range(8):
                value = int(round(float(clipped[row, col]) * 255))
                color = f"#{value:02x}{value:02x}{value:02x}"
                x0 = col * DEBUG_CELL_SIZE
                y0 = row * DEBUG_CELL_SIZE
                x1 = x0 + DEBUG_CELL_SIZE
                y1 = y0 + DEBUG_CELL_SIZE
                self.debug_canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=color,
                    outline="#555555",
                )

    def clear(self) -> None:
        self.canvas.delete("all")
        self.canvas_buffer.fill(255.0)
        self.last_preprocessing_stages = None
        self.last_prediction_result = None
        self.result_text.set("Draw a digit, then click Predict.")
        self.debug_text.set("Model input after preprocessing (8x8)\nNo prediction yet.")
        self.save_status_text.set("")
        self._draw_debug_grid(np.zeros((8, 8), dtype=float))

    def predict(self) -> None:
        stages = preprocess_canvas_image_with_stages(self.canvas_buffer)
        feature_vector = stages["feature_vector"]
        result = predict_single_digit(
            self.model,
            feature_vector,
            top_k=3,
        )

        self.last_preprocessing_stages = stages
        self.last_prediction_result = result
        self._draw_debug_grid(stages["resized_8x8"])

        predicted_index = int(result["prediction"])
        predicted_digit = self.class_names[predicted_index]
        lines = []
        if bool(stages["is_blank"]):
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

        lines.append(f"Checkpoint type: {self.checkpoint_type}")
        lines.append(f"Checkpoint path: {self.checkpoint_path}")
        lines.append(f"Foreground mass: {float(stages['foreground_mass']):.6f}")
        lines.append(f"Bounding box: {stages['bounding_box']}")
        self.result_text.set("\n".join(lines))

        warning = " blank input" if bool(stages["is_blank"]) else ""
        self.debug_text.set(
            (
                "Model input after preprocessing (8x8)\n"
                f"foreground_mass={float(stages['foreground_mass']):.6f}\n"
                f"bounding_box={stages['bounding_box']}\n"
                f"{warning}"
            ).strip()
        )

    def save_sample(self) -> None:
        if self.last_preprocessing_stages is None or self.last_prediction_result is None:
            self.save_status_text.set("Run prediction before saving a sample.")
            return

        label_text = self.true_label_entry.get().strip()
        try:
            true_label = None if label_text == "" else int(label_text)
            sample_path = save_canvas_sample(
                _project_path(DEFAULT_SAMPLE_DIR),
                self.canvas_buffer,
                self.last_preprocessing_stages,
                self.last_prediction_result,
                self.checkpoint_path,
                true_label=true_label,
            )
        except (TypeError, ValueError) as error:
            self.save_status_text.set(f"Could not save sample: {error}")
            return

        self.save_status_text.set(f"Saved sample: {sample_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a digit and classify it with the scratch MLP checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to a saved MulticlassMLPScratch checkpoint. "
            f"Default prefers {AUGMENTED_CHECKPOINT_PATH}, then falls back to "
            f"{BASELINE_CHECKPOINT_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path, checkpoint_type, warning = resolve_checkpoint_path(args.checkpoint)

    if checkpoint_path is None:
        print(
            "No checkpoint found. Run python experiments/compare_digits_augmented_training.py "
            "to create the augmented checkpoint.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not checkpoint_path.exists():
        print(
            f"Checkpoint not found: {checkpoint_path}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if warning is not None:
        print(warning, file=sys.stderr)

    root = tk.Tk()
    DigitDrawApp(
        root,
        checkpoint_path=checkpoint_path,
        checkpoint_type=checkpoint_type,
        checkpoint_warning=warning,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
