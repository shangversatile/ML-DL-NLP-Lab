"""Generate original T3 Learning From Data fitting/selection figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent


def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#334155",
    lw: float = 1.6,
    style: str = "->",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=14,
            linewidth=lw,
            color=color,
        )
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#1f2937",
    fontsize: int = 10,
) -> None:
    ax.add_patch(
        Rectangle(
            xy,
            width,
            height,
            linewidth=1.3,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        wrap=True,
    )


def logistic_score_probability_decision() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.8)

    _box(ax, (0.55, 2.1), 1.9, 0.9, "input\nx", "#dbeafe")
    _box(ax, (3.0, 2.1), 2.0, 0.9, "score\ns=w^T x", "#e0e7ff")
    _box(ax, (5.65, 2.1), 2.15, 0.9, "probability\np=sigmoid(s)", "#dcfce7")
    _box(ax, (8.55, 2.1), 2.25, 0.9, "decision\n1{p >= tau}", "#fef3c7")
    _arrow(ax, (2.45, 2.55), (3.0, 2.55))
    _arrow(ax, (5.0, 2.55), (5.65, 2.55))
    _arrow(ax, (7.8, 2.55), (8.55, 2.55))
    ax.text(4.0, 1.35, "parameterized linear evidence", ha="center", fontsize=9, color="#334155")
    ax.text(6.7, 1.35, "probabilistic model", ha="center", fontsize=9, color="#166534")
    ax.text(9.7, 1.35, "deployment rule / utility", ha="center", fontsize=9, color="#92400e")
    ax.set_title("Logistic regression separates score, probability, and decision", fontsize=14, weight="bold")
    _save(fig, "logistic_score_probability_decision.png")


def backpropagation_computational_graph() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)

    nodes = [
        ((0.6, 2.2), "x"),
        ((2.0, 2.2), "u=ax+b"),
        ((3.9, 2.2), "z=phi(u)"),
        ((5.9, 2.2), "r=cz+d"),
        ((8.0, 2.2), "L=ell(r,y)"),
    ]
    for xy, text in nodes:
        _box(ax, xy, 1.25 if text == "x" else 1.55, 0.75, text, "#e0f2fe")
    for i in range(len(nodes) - 1):
        start_x = nodes[i][0][0] + (1.25 if nodes[i][1] == "x" else 1.55)
        _arrow(ax, (start_x, 2.58), (nodes[i + 1][0][0], 2.58), "#2563eb", 1.6)

    backward = [
        ((8.0, 1.0), "dL/dr"),
        ((5.9, 1.0), "dL/dz"),
        ((3.9, 1.0), "dL/du"),
        ((2.0, 1.0), "dL/da,\ndL/db"),
    ]
    for xy, text in backward:
        _box(ax, xy, 1.55, 0.75, text, "#fee2e2", fontsize=9)
    _arrow(ax, (8.0, 1.38), (7.45, 2.2), "#dc2626", 1.4)
    _arrow(ax, (7.45, 1.38), (5.9, 1.38), "#dc2626", 1.5)
    _arrow(ax, (5.9, 1.38), (5.45, 2.2), "#dc2626", 1.4)
    _arrow(ax, (5.45, 1.38), (3.9, 1.38), "#dc2626", 1.5)
    _arrow(ax, (3.9, 1.38), (3.55, 2.2), "#dc2626", 1.4)
    _arrow(ax, (3.55, 1.38), (2.0, 1.38), "#dc2626", 1.5)
    ax.text(6.4, 4.25, "forward pass stores intermediate values", ha="center", fontsize=10, color="#1d4ed8")
    ax.text(5.2, 0.35, "backward pass reuses local derivatives", ha="center", fontsize=10, color="#991b1b")
    ax.set_title("Backpropagation is chain-rule derivative reuse", fontsize=14, weight="bold")
    _save(fig, "backpropagation_computational_graph.png")


def fixed_vs_learned_representation() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.4))
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 2.4)

    axes[0].set_title("Fixed feature transform", fontsize=13, weight="bold")
    _box(axes[0], (0.6, 0.85), 1.5, 0.7, "x", "#dbeafe")
    _box(axes[0], (3.0, 0.85), 2.0, 0.7, "fixed Phi(x)", "#e0e7ff")
    _box(axes[0], (6.0, 0.85), 2.0, 0.7, "learn w", "#dcfce7")
    _box(axes[0], (9.0, 0.85), 1.4, 0.7, "output", "#fef3c7")
    for a, b in [((2.1, 1.2), (3.0, 1.2)), ((5.0, 1.2), (6.0, 1.2)), ((8.0, 1.2), (9.0, 1.2))]:
        _arrow(axes[0], a, b)

    axes[1].set_title("Learned representation", fontsize=13, weight="bold")
    _box(axes[1], (0.6, 0.85), 1.5, 0.7, "x", "#dbeafe")
    _box(axes[1], (2.8, 0.85), 2.25, 0.7, "learn Phi_theta(x)", "#fde68a")
    _box(axes[1], (6.0, 0.85), 2.0, 0.7, "learn output", "#dcfce7")
    _box(axes[1], (9.0, 0.85), 1.4, 0.7, "output", "#fef3c7")
    for a, b in [((2.1, 1.2), (2.8, 1.2)), ((5.05, 1.2), (6.0, 1.2)), ((8.0, 1.2), (9.0, 1.2))]:
        _arrow(axes[1], a, b)
    axes[1].text(4.0, 0.25, "representation and predictor are fitted jointly", ha="center", fontsize=9, color="#92400e")

    fig.suptitle("Representation can be externally fixed or learned by the model", fontsize=14, weight="bold")
    _save(fig, "fixed_vs_learned_representation.png")


def overfitting_signal_noise_complexity() -> None:
    rng = np.random.default_rng(9)
    x = np.linspace(0, 1, 160)
    signal = 0.45 + 0.32 * np.sin(2 * np.pi * x)
    noise = rng.normal(0, 0.06, size=x.shape)
    train = signal + noise
    low = 0.45 + 0.25 * np.sin(2 * np.pi * x)
    high = train + 0.04 * np.sin(18 * np.pi * x)

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.scatter(x[::5], train[::5], s=20, color="#64748b", alpha=0.7, label="finite noisy sample")
    ax.plot(x, signal, color="#16a34a", lw=2.4, label="population signal")
    ax.plot(x, low, color="#2563eb", lw=2.0, label="lower-flexibility fit")
    ax.plot(x, high, color="#dc2626", lw=1.8, label="over-flexible fit")
    ax.set_title("Flexible fitting can capture signal and sample-specific noise", fontsize=14, weight="bold")
    ax.set_xlabel("input")
    ax.set_ylabel("target / prediction")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.53, 0.18, "low training error is not the definition;\nnoise-dependent behavior is the risk", fontsize=10, color="#7f1d1d")
    _save(fig, "overfitting_signal_noise_complexity.png")


def regularization_hard_soft_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

    theta = np.linspace(0, 2 * np.pi, 400)
    center = np.array([1.45, 0.9])
    for scale in [0.6, 1.2, 1.8, 2.4]:
        axes[0].plot(center[0] + scale * np.cos(theta), center[1] + 0.55 * scale * np.sin(theta), color="#94a3b8", lw=1)
    axes[0].plot(np.cos(theta) * 1.25, np.sin(theta) * 1.25, color="#2563eb", lw=2.1)
    axes[0].scatter([0.9], [0.85], color="#dc2626", s=70)
    axes[0].set_title("Hard constraint", fontsize=13, weight="bold")
    axes[0].text(0, -2.25, "minimize empirical loss\nsubject to Omega(w) <= C", ha="center", fontsize=10)

    for scale in [0.6, 1.1, 1.6, 2.1]:
        axes[1].plot(center[0] + scale * np.cos(theta), center[1] + 0.55 * scale * np.sin(theta), color="#94a3b8", lw=1)
    for scale in [0.45, 0.9, 1.35, 1.8]:
        axes[1].plot(scale * np.cos(theta), scale * np.sin(theta), color="#60a5fa", lw=0.9, alpha=0.7)
    axes[1].scatter([0.7], [0.55], color="#dc2626", s=70)
    axes[1].set_title("Soft penalty", fontsize=13, weight="bold")
    axes[1].text(0, -2.25, "minimize empirical loss\n+ lambda Omega(w)", ha="center", fontsize=10)

    fig.suptitle("Regularization changes feasibility or solution preference", fontsize=14, weight="bold")
    _save(fig, "regularization_hard_soft_geometry.png")


def regularization_solution_preference() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5.6)

    ax.add_patch(Rectangle((0.8, 1.2), 4.1, 3.0, facecolor="#eff6ff", edgecolor="#2563eb", lw=1.6))
    ax.text(2.85, 4.45, "many training-fitting solutions", ha="center", fontsize=12, weight="bold")
    positions = [(1.5, 2.0), (2.3, 3.1), (3.25, 2.4), (4.1, 3.4), (3.7, 1.7)]
    for idx, pos in enumerate(positions):
        color = "#16a34a" if idx == 2 else "#64748b"
        ax.add_patch(Circle(pos, 0.18, facecolor=color, edgecolor="#111827"))
    _box(ax, (6.35, 2.35), 2.8, 0.85, "regularizer / preference\nOmega or trajectory", "#fef3c7", fontsize=10)
    _arrow(ax, (4.9, 2.75), (6.35, 2.75), "#334155")
    _arrow(ax, (7.75, 2.35), (3.25, 2.4), "#16a34a", 1.8)
    ax.text(7.75, 1.35, "selection is biased toward\npreferred solution type", ha="center", fontsize=10, color="#166534")
    ax.set_title("Regularization supplies inductive bias among compatible fits", fontsize=14, weight="bold")
    _save(fig, "regularization_solution_preference.png")


def validation_selection_dependency() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.4)

    _box(ax, (0.6, 3.6), 1.9, 0.75, "train set", "#dbeafe")
    candidates = [(3.2, 4.2, "g1"), (3.2, 3.2, "g2"), (3.2, 2.2, "g3")]
    for x, y, t in candidates:
        _box(ax, (x, y), 1.25, 0.6, t, "#e0e7ff")
        _arrow(ax, (2.5, 3.98), (x, y + 0.3), "#64748b", 1.1)

    _box(ax, (5.6, 3.0), 2.0, 0.8, "validation set\nestimates risks", "#dcfce7")
    for _, y, _ in candidates:
        _arrow(ax, (4.45, y + 0.3), (5.6, 3.4), "#334155", 1.1)
    _box(ax, (8.8, 3.0), 1.8, 0.8, "select min\nval error", "#fef3c7")
    _arrow(ax, (7.6, 3.4), (8.8, 3.4))
    _box(ax, (8.95, 1.55), 1.5, 0.65, "g_hat_m", "#fee2e2")
    _arrow(ax, (9.7, 3.0), (9.7, 2.2), "#dc2626", 1.8)
    ax.text(6.5, 1.15, "after selection, final candidate depends on validation data", ha="center", fontsize=10, color="#991b1b")
    ax.set_title("Validation is evaluation before selection and selection signal after reuse", fontsize=14, weight="bold")
    _save(fig, "validation_selection_dependency.png")


def adaptive_validation_contamination_loop() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 5.8)

    _box(ax, (0.7, 3.7), 2.0, 0.75, "candidate\nprocedure", "#e0e7ff")
    _box(ax, (3.6, 3.7), 1.8, 0.75, "train", "#dbeafe")
    _box(ax, (6.2, 3.7), 2.0, 0.75, "validation\nfeedback", "#dcfce7")
    _box(ax, (8.9, 3.7), 2.0, 0.75, "revise\ncandidates", "#fef3c7")
    _arrow(ax, (2.7, 4.08), (3.6, 4.08))
    _arrow(ax, (5.4, 4.08), (6.2, 4.08))
    _arrow(ax, (8.2, 4.08), (8.9, 4.08))
    _arrow(ax, (9.9, 3.7), (1.7, 3.7), "#dc2626", 1.6)
    ax.text(5.8, 2.95, "adaptive loop: evaluation outcome changes future search", ha="center", fontsize=10, color="#991b1b")
    _box(ax, (4.6, 1.05), 2.6, 0.8, "final model is selected\nby the whole loop", "#fee2e2")
    _arrow(ax, (7.1, 3.7), (5.9, 1.85), "#dc2626", 1.4)
    ax.set_title("Repeated validation feedback contaminates the development signal", fontsize=14, weight="bold")
    _save(fig, "adaptive_validation_contamination_loop.png")


def train_val_test_information_flow() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.4)

    _box(ax, (0.7, 3.7), 2.0, 0.75, "D_train", "#dbeafe")
    _box(ax, (3.7, 3.7), 2.25, 0.75, "fit parameters", "#e0e7ff")
    _box(ax, (0.7, 2.35), 2.0, 0.75, "D_val / dev", "#dcfce7")
    _box(ax, (3.7, 2.35), 2.25, 0.75, "select model /\nhyperparameters", "#fef3c7")
    _box(ax, (6.9, 3.0), 2.2, 0.85, "frozen final\nprocedure", "#ede9fe")
    _box(ax, (9.8, 3.0), 1.9, 0.85, "D_test\nfinal eval", "#fee2e2")

    _arrow(ax, (2.7, 4.08), (3.7, 4.08))
    _arrow(ax, (2.7, 2.73), (3.7, 2.73))
    _arrow(ax, (5.95, 4.08), (6.9, 3.5))
    _arrow(ax, (5.95, 2.73), (6.9, 3.35))
    _arrow(ax, (9.1, 3.43), (9.8, 3.43))
    _arrow(ax, (10.75, 3.0), (6.95, 2.15), "#dc2626", 1.3)
    ax.text(8.65, 1.75, "forbidden if D_test is final evidence:\nfeedback into design", ha="center", fontsize=9.5, color="#991b1b")
    ax.text(9.9, 4.35, "one-way evaluation flow", ha="center", fontsize=9.5, color="#334155")
    ax.set_title("Dataset roles are defined by allowed information flow", fontsize=14, weight="bold")
    _save(fig, "train_val_test_information_flow.png")


def selection_aware_research_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    ax.set_axis_off()
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.2)

    boxes = [
        ((0.55, 4.6), "research\nquestion", "#e0f2fe"),
        ((2.3, 4.6), "dataset\nchoice", "#dbeafe"),
        ((4.05, 4.6), "preprocess", "#e0e7ff"),
        ((5.8, 4.6), "model\nfamily", "#ede9fe"),
        ((7.55, 4.6), "training", "#dcfce7"),
        ((9.3, 4.6), "validation\nfeedback", "#fef3c7"),
        ((9.3, 2.95), "revise\nprocedure", "#fee2e2"),
        ((5.0, 1.25), "freeze final\nprocedure", "#fce7f3"),
        ((9.3, 1.25), "independent\nevaluation", "#cffafe"),
    ]
    for xy, text, color in boxes:
        _box(ax, xy, 1.35, 0.8, text, color, fontsize=9)
    for i in range(5):
        _arrow(ax, (boxes[i][0][0] + 1.35, 5.0), (boxes[i + 1][0][0], 5.0))
    _arrow(ax, (8.9, 5.0), (9.3, 5.0))
    _arrow(ax, (10.0, 4.6), (10.0, 3.75), "#dc2626")
    _arrow(ax, (9.3, 3.35), (5.8, 4.6), "#dc2626", 1.4)
    ax.text(7.6, 3.45, "feedback changes future candidates", ha="center", fontsize=9, color="#991b1b")
    _arrow(ax, (7.55, 4.6), (5.9, 2.05), "#334155")
    _arrow(ax, (6.35, 1.65), (9.3, 1.65), "#334155")
    ax.plot([8.05, 8.05], [0.5, 2.45], color="#111827", linestyle="--", lw=1.3)
    ax.text(8.05, 0.55, "freeze point", ha="center", fontsize=10, weight="bold")
    ax.set_title("Selection-aware research protocol tracks the whole adaptive path", fontsize=14, weight="bold")
    _save(fig, "selection_aware_research_pipeline.png")


def main() -> None:
    logistic_score_probability_decision()
    backpropagation_computational_graph()
    fixed_vs_learned_representation()
    overfitting_signal_noise_complexity()
    regularization_hard_soft_geometry()
    regularization_solution_preference()
    validation_selection_dependency()
    adaptive_validation_contamination_loop()
    train_val_test_information_flow()
    selection_aware_research_pipeline()


if __name__ == "__main__":
    main()
