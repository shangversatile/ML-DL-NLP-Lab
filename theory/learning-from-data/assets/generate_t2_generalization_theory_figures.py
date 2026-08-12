"""Generate original T2 Learning From Data generalization-theory figures."""

from __future__ import annotations

from math import comb
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
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
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
            linewidth=1.4,
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


def training_vs_testing_dependency() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.8)

    ax.text(2.9, 5.25, "Testing: fixed g relative to test data", ha="center", fontsize=13, weight="bold")
    ax.text(9.7, 5.25, "Training: g selected by same data", ha="center", fontsize=13, weight="bold")

    _box(ax, (0.55, 3.35), 2.0, 0.78, "training data", "#dbeafe")
    _box(ax, (3.05, 3.35), 1.55, 0.78, "choose g", "#ede9fe")
    _box(ax, (5.05, 3.35), 1.75, 0.78, "fixed g", "#ffe4e6")
    _arrow(ax, (2.55, 3.74), (3.05, 3.74))
    _arrow(ax, (4.6, 3.74), (5.05, 3.74))

    _box(ax, (0.55, 1.6), 2.0, 0.78, "independent\ntest data", "#dcfce7")
    _box(ax, (3.05, 1.6), 1.55, 0.78, "evaluate", "#fef3c7")
    _box(ax, (5.05, 1.6), 1.75, 0.78, "E_test(g)", "#fae8ff")
    _arrow(ax, (2.55, 1.99), (3.05, 1.99))
    _arrow(ax, (4.6, 1.99), (5.05, 1.99))
    _arrow(ax, (5.93, 3.35), (3.83, 2.38), "#64748b", 1.2)
    ax.text(4.9, 2.65, "g fixed before evaluation", ha="center", fontsize=9, color="#334155")

    _box(ax, (7.4, 2.9), 2.1, 0.85, "same training\nset D", "#dbeafe")
    _box(ax, (10.1, 3.55), 1.9, 0.8, "select\ng=A(D)", "#ede9fe")
    _box(ax, (10.1, 2.05), 1.9, 0.8, "evaluate\nE_in(g)", "#fef3c7")
    _arrow(ax, (9.5, 3.32), (10.1, 3.9))
    _arrow(ax, (9.5, 3.32), (10.1, 2.45))
    _arrow(ax, (11.05, 3.55), (11.05, 2.85), "#ef4444", 1.8)
    ax.text(10.9, 1.35, "adaptive dependence:\nD changes -> selected g changes", ha="center", fontsize=9.5, color="#991b1b")

    ax.plot([6.95, 6.95], [0.7, 5.25], color="#94a3b8", lw=1.3, linestyle="--")
    _save(fig, "training_vs_testing_dependency.png")


def uniform_convergence_envelope() -> None:
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    x = np.linspace(0, 1, 160)
    base = 0.22 + 0.62 * x
    eps = 0.08 + 0.035 * np.sin(2 * np.pi * x)
    ax.fill_between(x, base - eps, base + eps, color="#bfdbfe", alpha=0.75, label="uniform envelope")
    ax.plot(x, base, color="#1d4ed8", lw=2.2, label="population risk R(h)")

    for i in range(11):
        jitter = rng.normal(0, 0.018, size=len(x))
        empirical = base + 0.045 * np.sin((i + 2) * np.pi * x + i * 0.35) + jitter * 0.25
        ax.plot(x, empirical, color="#475569", lw=0.8, alpha=0.35)

    selected_x = 0.72
    selected_y = 0.22 + 0.62 * selected_x
    ax.scatter([selected_x], [selected_y], s=95, color="#e11d48", zorder=5, label="selected g")
    ax.vlines(selected_x, selected_y - 0.11, selected_y + 0.11, color="#e11d48", lw=1.5, linestyle="--")
    ax.text(selected_x + 0.025, selected_y + 0.09, "data-selected g\ncovered by sup bound", fontsize=9, color="#881337")

    ax.set_title("Uniform convergence controls all candidate hypotheses simultaneously", fontsize=14, weight="bold")
    ax.set_xlabel("hypotheses ordered only for visualization")
    ax.set_ylabel("risk / empirical risk")
    ax.set_ylim(0.05, 0.95)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    _save(fig, "uniform_convergence_envelope.png")


def growth_function_breakpoint_vc() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    n = np.arange(1, 16)
    maximal = 2**n
    d = 3
    polynomial = np.array([sum(comb(int(k), i) for i in range(d + 1)) for k in n])
    ax.semilogy(n, maximal, color="#ef4444", lw=2.2, label="maximal dichotomies 2^N")
    ax.semilogy(n, polynomial, color="#2563eb", lw=2.2, label="VC-controlled growth, d_VC=3")
    ax.axvline(d + 1, color="#111827", linestyle="--", lw=1.4)
    ax.text(d + 1 + 0.25, maximal[d] * 0.5, "break point k=4", fontsize=10, color="#111827")
    ax.fill_between(n[n <= d], maximal[n <= d] * 0.75, maximal[n <= d] * 1.25, color="#fee2e2", alpha=0.35)
    ax.set_title("Finite break point changes growth from exponential to controlled", fontsize=14, weight="bold")
    ax.set_xlabel("number of points N")
    ax.set_ylabel("number of dichotomies (log scale)")
    ax.set_xticks(n)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "growth_function_breakpoint_vc.png")


def vc_dimension_shattering_geometry() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))
    for ax in axes:
        ax.set_axis_off()

    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Positive rays: d_VC=1", fontsize=12, weight="bold")
    ax.hlines(0.55, 0.08, 0.92, color="#334155", lw=2)
    for x, label, color in [(0.34, "+", "#16a34a"), (0.70, "-", "#dc2626")]:
        ax.add_patch(Circle((x, 0.55), 0.045, facecolor=color, edgecolor="#111827"))
        ax.text(x, 0.75, label, ha="center", fontsize=13)
    ax.vlines(0.50, 0.35, 0.75, color="#2563eb", lw=2)
    ax.text(0.5, 0.22, "pattern (+,-) impossible\nfor h_a(x)=1{x>=a}", ha="center", fontsize=9)

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Intervals: d_VC=2", fontsize=12, weight="bold")
    ax.hlines(0.55, 0.08, 0.92, color="#334155", lw=2)
    xs = [0.24, 0.50, 0.76]
    labels = ["+", "-", "+"]
    colors = ["#16a34a", "#dc2626", "#16a34a"]
    for x, label, color in zip(xs, labels, colors):
        ax.add_patch(Circle((x, 0.55), 0.045, facecolor=color, edgecolor="#111827"))
        ax.text(x, 0.75, label, ha="center", fontsize=13)
    ax.add_patch(Rectangle((0.19, 0.42), 0.62, 0.26, facecolor="none", edgecolor="#2563eb", lw=2))
    ax.text(0.5, 0.22, "pattern (+,-,+) impossible\nfor one contiguous interval", ha="center", fontsize=9)

    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("2D linear separators: d_VC=3", fontsize=12, weight="bold")
    pts = np.array([[0.22, 0.25], [0.78, 0.28], [0.80, 0.78], [0.24, 0.75]])
    cols = ["#16a34a", "#dc2626", "#16a34a", "#dc2626"]
    labs = ["+", "-", "+", "-"]
    for (x, y), color, label in zip(pts, cols, labs):
        ax.add_patch(Circle((x, y), 0.045, facecolor=color, edgecolor="#111827"))
        ax.text(x, y + 0.095, label, ha="center", fontsize=13)
    ax.plot([pts[0, 0], pts[1, 0], pts[2, 0], pts[3, 0], pts[0, 0]], [pts[0, 1], pts[1, 1], pts[2, 1], pts[3, 1], pts[0, 1]], color="#94a3b8", lw=1.3)
    ax.text(0.5, 0.08, "alternating labels on a convex quadrilateral\ncannot be separated by one line", ha="center", fontsize=9)

    fig.suptitle("Shattering asks which labelings can be realized on finite point sets", fontsize=14, weight="bold")
    _save(fig, "vc_dimension_shattering_geometry.png")


def generalization_bound_capacity_sample_size() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    n = np.linspace(20, 2000, 250)
    capacities = [5, 20, 80]
    colors = ["#16a34a", "#2563eb", "#dc2626"]
    for d, color in zip(capacities, colors):
        width = np.sqrt((d * np.log(np.maximum(n / d, 1.05)) + np.log(40)) / n)
        ax.plot(n, width, color=color, lw=2.2, label=f"capacity d={d}")
    ax.set_title("Generalization-bound width shrinks with N and grows with capacity", fontsize=14, weight="bold")
    ax.set_xlabel("sample size N")
    ax.set_ylabel("conceptual bound width")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(1180, 0.82, "same confidence level\nsame bounded-loss setup", fontsize=10, color="#334155")
    _save(fig, "generalization_bound_capacity_sample_size.png")


def bias_variance_dataset_randomness() -> None:
    rng = np.random.default_rng(13)
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    x = np.linspace(-2.5, 2.5, 220)
    target = 0.4 * x**2 - 0.25
    preds = []
    for _ in range(11):
        a = 0.4 + rng.normal(0, 0.045)
        b = rng.normal(0, 0.12)
        c = -0.25 + rng.normal(0, 0.10)
        y = a * x**2 + b * x + c
        preds.append(y)
        ax.plot(x, y, color="#64748b", alpha=0.35, lw=1.1)
    avg = np.mean(preds, axis=0)
    ax.plot(x, target, color="#111827", lw=2.4, label="target f(x)")
    ax.plot(x, avg, color="#2563eb", lw=2.4, label="average hypothesis g_bar(x)")
    x0 = 1.45
    idx = np.argmin(np.abs(x - x0))
    ax.vlines(x0, target[idx], avg[idx], color="#dc2626", lw=2, label="bias at x")
    vals = [p[idx] for p in preds]
    ax.scatter([x0] * len(vals), vals, color="#f97316", s=24, zorder=4, label="dataset-specific g_D(x)")
    ax.set_title("Different random datasets induce different learned hypotheses", fontsize=14, weight="bold")
    ax.set_xlabel("input x")
    ax.set_ylabel("prediction")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "bias_variance_dataset_randomness.png")


def learning_curves_bias_variance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), sharey=True)
    n = np.linspace(20, 1000, 160)

    train_bias = 0.42 - 0.08 * (1 - np.exp(-n / 220))
    test_bias = 0.48 - 0.09 * (1 - np.exp(-n / 260))
    train_var = 0.06 + 0.16 * np.exp(-n / 180)
    test_var = 0.14 + 0.46 * np.exp(-n / 260)

    configs = [
        (axes[0], train_bias, test_bias, "High-bias regime", "gap small, both errors high"),
        (axes[1], train_var, test_var, "High-variance regime", "gap large, more data helps"),
    ]
    for ax, train, test, title, subtitle in configs:
        ax.plot(n, train, color="#2563eb", lw=2.4, label="training error")
        ax.plot(n, test, color="#dc2626", lw=2.4, label="out-of-sample error")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.text(0.5, 0.91, subtitle, transform=ax.transAxes, ha="center", fontsize=10, color="#334155")
        ax.set_xlabel("sample size N")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("error")
    fig.suptitle("Learning curves diagnose regimes but do not prove assumptions", fontsize=14, weight="bold")
    _save(fig, "learning_curves_bias_variance.png")


def error_source_decomposition_map() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 5.4)

    stages = [
        ("World / target\nand stochastic P", "#dbeafe"),
        ("Measurement\nmechanism", "#e0f2fe"),
        ("Representation\nPhi(x)", "#dcfce7"),
        ("Hypothesis\nfamily H_Phi", "#fef3c7"),
        ("Finite sample\nselection", "#ede9fe"),
        ("Optimization\nalgorithm", "#fae8ff"),
        ("Selected\nhypothesis g", "#ffe4e6"),
    ]
    xs = [0.25, 2.15, 4.05, 5.95, 7.85, 9.75, 11.65]
    for (label, color), x0 in zip(stages, xs):
        _box(ax, (x0, 2.7), 1.55, 0.85, label, color, fontsize=9)
    for x0 in xs[:-1]:
        _arrow(ax, (x0 + 1.55, 3.12), (x0 + 1.9, 3.12))

    failures = [
        (1.0, "irreducible\nstochastic\nuncertainty", "#bfdbfe"),
        (3.15, "information /\nrepresentation\nfailure", "#bbf7d0"),
        (5.85, "approximation /\nspecification\nerror", "#fde68a"),
        (8.05, "estimation /\ngeneralization\nerror", "#ddd6fe"),
        (10.35, "optimization /\ncomputation\nerror", "#f5d0fe"),
    ]
    for x0, label, color in failures:
        _box(ax, (x0, 0.9), 1.9, 0.92, label, color, "#334155", fontsize=9)
        _arrow(ax, (x0 + 0.95, 2.7), (x0 + 0.95, 1.82), "#64748b", 1.2)

    ax.text(6.9, 4.65, "Different error sources answer different research questions", ha="center", fontsize=14, weight="bold")
    ax.text(6.9, 0.25, "Do not collapse representation loss, class misspecification, finite-sample selection, optimization failure, and noise into one overfitting label.", ha="center", fontsize=10, color="#334155")
    _save(fig, "error_source_decomposition_map.png")


def main() -> None:
    training_vs_testing_dependency()
    uniform_convergence_envelope()
    growth_function_breakpoint_vc()
    vc_dimension_shattering_geometry()
    generalization_bound_capacity_sample_size()
    bias_variance_dataset_randomness()
    learning_curves_bias_variance()
    error_source_decomposition_map()


if __name__ == "__main__":
    main()
