"""Generate original T1 Learning From Data theory figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent


def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.6,
            color="#334155",
        )
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
) -> None:
    ax.add_patch(
        Rectangle(
            xy,
            width,
            height,
            linewidth=1.4,
            edgecolor="#1f2937",
            facecolor=facecolor,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="#111827",
        wrap=True,
    )


def learning_system_world_to_hypothesis() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 5)

    items = [
        ("Unknown\nworld / target", "#dbeafe"),
        ("Sampling\nprocess", "#e0f2fe"),
        ("Finite\ndataset D", "#dcfce7"),
        ("Feature\nrepresentation\nPhi(x)", "#fef3c7"),
        ("Hypothesis\nset H", "#ede9fe"),
        ("Learning\nalgorithm A", "#fae8ff"),
        ("Selected\nhypothesis g", "#ffe4e6"),
    ]
    xs = np.linspace(0.35, 11.75, len(items))
    for i, ((label, color), x) in enumerate(zip(items, xs)):
        _box(ax, (x, 2.25), 1.35, 1.0, label, color)
        if i < len(items) - 1:
            _arrow(ax, (x + 1.35, 2.75), (xs[i + 1], 2.75))

    _box(ax, (11.0, 0.55), 1.75, 0.8, "Out-of-sample\nevaluation", "#f1f5f9")
    _arrow(ax, (12.55, 2.25), (12.0, 1.35))
    ax.text(
        6,
        4.25,
        "The learner sees represented finite evidence, not the world directly.",
        ha="center",
        fontsize=13,
        weight="bold",
        color="#0f172a",
    )
    ax.text(
        6,
        0.15,
        "Each arrow can add assumptions, noise, information loss, or selection bias.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    _save(fig, "learning_system_world_to_hypothesis.png")


def finite_sample_generalization_bridge() -> None:
    rng = np.random.default_rng(7)
    sample_sizes = np.arange(10, 501, 10)
    true_error = 0.28
    trials = 350
    means = []
    lower = []
    upper = []
    for n in sample_sizes:
        draws = rng.binomial(n, true_error, size=trials) / n
        means.append(draws.mean())
        lower.append(np.quantile(draws, 0.05))
        upper.append(np.quantile(draws, 0.95))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(sample_sizes, means, color="#2563eb", linewidth=2, label="mean empirical error")
    ax.fill_between(sample_sizes, lower, upper, color="#93c5fd", alpha=0.45, label="5%-95% sample range")
    ax.axhline(true_error, color="#b91c1c", linestyle="--", linewidth=2, label="population error")
    ax.set_title("Concentration links finite samples to population error", fontsize=14, weight="bold")
    ax.set_xlabel("sample size N")
    ax.set_ylabel("error")
    ax.set_ylim(0.0, 0.6)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        260,
        0.08,
        "fixed hypothesis h\nempirical average stabilizes as N grows",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )
    _save(fig, "finite_sample_generalization_bridge.png")


def hypothesis_space_and_selection() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    ax.add_patch(Ellipse((5, 3.5), 6.8, 4.5, facecolor="#eef2ff", edgecolor="#4338ca", linewidth=2))
    ax.text(5, 6.15, "hypothesis set H", ha="center", fontsize=14, weight="bold", color="#312e81")

    points = {
        "h_bad": (2.8, 3.0, "#ef4444", "low Ein\nhigh Eout"),
        "h_star": (6.8, 4.35, "#16a34a", "best in H\nh*"),
        "g": (5.1, 2.25, "#2563eb", "selected\ng=A(D)"),
    }
    for _, (x, y, color, label) in points.items():
        ax.add_patch(Circle((x, y), 0.16, facecolor=color, edgecolor="#111827", linewidth=1))
        ax.text(x + 0.25, y + 0.05, label, fontsize=10, va="center", color="#111827")

    ax.add_patch(Circle((8.7, 5.55), 0.18, facecolor="#f97316", edgecolor="#111827", linewidth=1))
    ax.text(7.85, 6.0, "target f\noutside H", fontsize=10, color="#111827")
    _arrow(ax, (8.55, 5.45), (7.35, 4.6))
    ax.text(7.4, 4.85, "representation gap", fontsize=10, color="#9a3412")

    _arrow(ax, (3.1, 2.95), (4.9, 2.35))
    ax.text(3.0, 2.25, "sample-driven selection\ncan chase noise", fontsize=10, color="#991b1b")

    _arrow(ax, (5.25, 2.4), (6.65, 4.15))
    ax.text(5.55, 3.4, "estimation /\ngeneralization gap", fontsize=10, color="#1d4ed8")

    ax.text(
        5,
        0.65,
        "Learning succeeds only when representation, estimation, and optimization align.",
        ha="center",
        fontsize=12,
        weight="bold",
        color="#0f172a",
    )
    _save(fig, "hypothesis_space_and_selection.png")


def linear_feature_transform_geometry() -> None:
    rng = np.random.default_rng(4)
    n = 110
    angles = rng.uniform(0, 2 * np.pi, n)
    inner_r = rng.normal(0.55, 0.08, n)
    outer_r = rng.normal(1.25, 0.10, n)
    inner = np.c_[inner_r * np.cos(angles), inner_r * np.sin(angles)]
    outer = np.c_[outer_r * np.cos(angles), outer_r * np.sin(angles)]

    phi_inner = np.c_[inner[:, 0], inner[:, 0] ** 2 + inner[:, 1] ** 2]
    phi_outer = np.c_[outer[:, 0], outer[:, 0] ** 2 + outer[:, 1] ** 2]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6))
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].scatter(inner[:, 0], inner[:, 1], s=22, color="#2563eb", label="class -1")
    axes[0].scatter(outer[:, 0], outer[:, 1], s=22, color="#dc2626", label="class +1")
    axes[0].set_title("Original input space x", fontsize=11.5, weight="bold", pad=8)
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(frameon=False)
    axes[0].text(
        -1.42,
        -1.38,
        "No straight line\nseparates rings",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )

    axes[1].scatter(phi_inner[:, 0], phi_inner[:, 1], s=22, color="#2563eb", label="Phi(inner)")
    axes[1].scatter(phi_outer[:, 0], phi_outer[:, 1], s=22, color="#dc2626", label="Phi(outer)")
    axes[1].axhline(0.95, color="#111827", linewidth=2, label="linear separator")
    axes[1].set_title(
        "Feature space\nPhi(x) = [x1, x1^2 + x2^2]",
        fontsize=11.5,
        weight="bold",
        pad=8,
    )
    axes[1].set_xlabel("z1")
    axes[1].set_ylabel("z2")
    axes[1].legend(frameon=False)
    axes[1].text(
        -1.45,
        2.15,
        "Linear in feature space\nnonlinear in original space",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )
    fig.suptitle("Feature transforms change hypothesis geometry", fontsize=14, weight="bold", y=0.98)
    fig.subplots_adjust(top=0.78, wspace=0.2)
    _save(fig, "linear_feature_transform_geometry.png")


def error_noise_target_model() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)

    _box(ax, (0.4, 3.9), 1.7, 0.85, "latent\nworld state", "#dbeafe")
    _box(ax, (2.8, 3.9), 1.7, 0.85, "clean target\nor P(Y|X)", "#dcfce7")
    _box(ax, (5.2, 4.55), 1.7, 0.85, "label noise", "#fee2e2")
    _box(ax, (5.2, 3.25), 1.7, 0.85, "measurement\nnoise", "#ffedd5")
    _box(ax, (7.6, 3.9), 1.7, 0.85, "observed\n(x, y)", "#fef3c7")
    _box(ax, (4.0, 1.25), 1.7, 0.85, "loss l", "#ede9fe")
    _box(ax, (6.45, 1.25), 1.7, 0.85, "learning\nalgorithm", "#fae8ff")
    _box(ax, (8.85, 1.25), 1.7, 0.85, "selected\nmodel g", "#ffe4e6")

    _arrow(ax, (2.1, 4.32), (2.8, 4.32))
    _arrow(ax, (4.5, 4.32), (5.2, 4.92))
    _arrow(ax, (4.5, 4.32), (5.2, 3.68))
    _arrow(ax, (6.9, 4.92), (7.6, 4.32))
    _arrow(ax, (6.9, 3.68), (7.6, 4.32))
    _arrow(ax, (8.45, 3.9), (7.3, 2.1))
    _arrow(ax, (5.7, 1.68), (6.45, 1.68))
    _arrow(ax, (8.15, 1.68), (8.85, 1.68))

    ax.text(
        5.5,
        5.65,
        "Observed error mixes target uncertainty, noise, representation, and loss.",
        ha="center",
        fontsize=13,
        weight="bold",
        color="#0f172a",
    )
    ax.text(
        5.5,
        0.45,
        "Changing the loss or noise assumption can change the optimal hypothesis.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    _save(fig, "error_noise_target_model.png")


def main() -> None:
    learning_system_world_to_hypothesis()
    finite_sample_generalization_bridge()
    hypothesis_space_and_selection()
    linear_feature_transform_geometry()
    error_noise_target_model()


if __name__ == "__main__":
    main()
