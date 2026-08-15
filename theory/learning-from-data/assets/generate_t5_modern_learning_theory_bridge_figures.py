"""Generate original T5 modern learning theory bridge figures."""

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
    fontsize: int = 9,
) -> None:
    ax.add_patch(
        Rectangle(
            xy,
            width,
            height,
            linewidth=1.25,
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


def class_vs_algorithm_dependent_generalization() -> None:
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))

    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.add_patch(Circle((2.5, 2.5), 1.9, facecolor="#e0e7ff", edgecolor="#3730a3", lw=1.6))
    pts = rng.normal([2.5, 2.5], [0.85, 0.85], size=(90, 2))
    mask = np.linalg.norm(pts - np.array([2.5, 2.5]), axis=1) < 1.8
    pts = pts[mask]
    ax.scatter(pts[:, 0], pts[:, 1], s=22, color="#6366f1", alpha=0.65)
    ax.text(2.5, 4.65, "Class view", ha="center", fontsize=13, weight="bold")
    ax.text(2.5, 0.18, "control all h in H", ha="center", fontsize=10, color="#334155")
    ax.text(2.5, 2.5, "H", ha="center", va="center", fontsize=24, weight="bold", color="#312e81")

    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.add_patch(Circle((2.5, 2.5), 1.9, facecolor="#f1f5f9", edgecolor="#64748b", lw=1.3))
    pts = rng.normal([2.5, 2.5], [0.85, 0.85], size=(90, 2))
    mask = np.linalg.norm(pts - np.array([2.5, 2.5]), axis=1) < 1.8
    pts = pts[mask]
    ax.scatter(pts[:, 0], pts[:, 1], s=18, color="#cbd5e1", alpha=0.85)
    path = np.array([[0.9, 0.75], [1.45, 1.25], [2.0, 1.85], [2.7, 2.25], [3.35, 2.85]])
    ax.plot(path[:, 0], path[:, 1], color="#dc2626", lw=2.3)
    ax.scatter([path[-1, 0]], [path[-1, 1]], color="#dc2626", s=90, zorder=5)
    ax.text(2.5, 4.65, "Algorithm view", ha="center", fontsize=13, weight="bold")
    ax.text(2.5, 0.18, "study selected g = A(S)", ha="center", fontsize=10, color="#334155")
    ax.text(3.55, 3.05, "A(S)", fontsize=11, color="#991b1b", weight="bold")
    fig.suptitle("Class-dependent and algorithm-dependent generalization ask different questions", fontsize=14, weight="bold")
    _save(fig, "class_vs_algorithm_dependent_generalization.png")


def algorithmic_stability_neighboring_datasets() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.6)

    _box(ax, (0.6, 3.35), 2.0, 0.9, "S\none dataset", "#dbeafe")
    _box(ax, (0.6, 1.35), 2.0, 0.9, "S without i\nneighbor", "#e0f2fe")
    ax.text(1.6, 2.8, "change one example", ha="center", fontsize=9, color="#334155")
    _box(ax, (4.0, 3.35), 2.0, 0.9, "h_S = A(S)", "#dcfce7")
    _box(ax, (4.0, 1.35), 2.0, 0.9, "h_neighbor", "#dcfce7")
    _box(ax, (7.4, 3.35), 1.8, 0.9, "loss at z", "#fef3c7")
    _box(ax, (7.4, 1.35), 1.8, 0.9, "loss at z", "#fef3c7")
    _box(ax, (10.2, 2.35), 1.7, 0.9, "difference\n<= beta", "#fee2e2")
    _arrow(ax, (2.6, 3.8), (4.0, 3.8))
    _arrow(ax, (2.6, 1.8), (4.0, 1.8))
    _arrow(ax, (6.0, 3.8), (7.4, 3.8))
    _arrow(ax, (6.0, 1.8), (7.4, 1.8))
    _arrow(ax, (9.2, 3.8), (10.2, 3.0))
    _arrow(ax, (9.2, 1.8), (10.2, 2.7))
    ax.set_title("Algorithmic stability measures sensitivity to training-set perturbation", fontsize=14, weight="bold")
    _save(fig, "algorithmic_stability_neighboring_datasets.png")


def rademacher_random_sign_complexity() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(-2.6, 2.6, 32)
    y_signal = (x > 0).astype(float)
    signs = rng.choice([-1, 1], size=len(x))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    axes[0].scatter(x, y_signal, c=np.where(y_signal > 0, "#2563eb", "#dc2626"), s=42)
    axes[0].plot(x, 1 / (1 + np.exp(-5 * x)), color="#111827", lw=2)
    axes[0].set_title("structured labels", fontsize=12, weight="bold")
    axes[0].text(-2.45, 0.82, "simple rule can fit signal", fontsize=9, color="#334155")

    axes[1].scatter(x, signs, c=np.where(signs > 0, "#2563eb", "#dc2626"), s=42)
    axes[1].axhline(0, color="#111827", lw=1.2)
    axes[1].plot(x, 0.35 * np.sin(1.6 * x), color="#64748b", lw=2)
    axes[1].set_title("random signs", fontsize=12, weight="bold")
    axes[1].text(-2.45, 0.58, "rich class correlates more\nwith noise-like signs", fontsize=9, color="#334155")

    for ax in axes:
        ax.set_ylim(-1.35, 1.35)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("sample coordinate")
    axes[0].set_ylabel("label or sign")
    fig.suptitle("Rademacher complexity asks how well a class fits random signs on this sample", fontsize=14, weight="bold")
    _save(fig, "rademacher_random_sign_complexity.png")


def norm_bounded_rademacher_geometry() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.22)
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.7, 2.7)
    ax.add_patch(Circle((0, 0), 2.0, fill=False, edgecolor="#2563eb", lw=2.2))
    v = np.array([1.55, 0.95])
    ax.arrow(0, 0, v[0], v[1], width=0.025, head_width=0.16, color="#dc2626", length_includes_head=True)
    w = 2.0 * v / np.linalg.norm(v)
    ax.arrow(0, 0, w[0], w[1], width=0.018, head_width=0.13, color="#16a34a", length_includes_head=True)
    ax.text(0.1, 2.12, "||w|| <= B", fontsize=10, color="#1d4ed8")
    ax.text(v[0] + 0.08, v[1] + 0.08, "sum sigma_i x_i", fontsize=9, color="#991b1b")
    ax.text(w[0] - 0.1, w[1] + 0.22, "best aligned w", fontsize=9, color="#166534")
    ax.set_title("Norm bound limits alignment with random signed sample vector", fontsize=14, weight="bold")
    ax.set_xlabel("coordinate 1")
    ax.set_ylabel("coordinate 2")
    _save(fig, "norm_bounded_rademacher_geometry.png")


def interpolation_double_descent_regimes() -> None:
    x = np.linspace(0.25, 3.2, 450)
    curve = 0.42 + 0.4 / (x + 0.25) + 0.95 * np.exp(-((x - 1.15) / 0.18) ** 2) + 0.12 / (x + 0.2)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.plot(x, curve, color="#111827", lw=2.5)
    ax.axvline(1.15, color="#dc2626", lw=1.8, linestyle="--")
    ax.axvspan(0.25, 1.0, color="#dbeafe", alpha=0.45)
    ax.axvspan(1.0, 1.32, color="#fee2e2", alpha=0.55)
    ax.axvspan(1.32, 3.2, color="#dcfce7", alpha=0.45)
    ax.text(0.48, 1.42, "underparameterized", fontsize=10, color="#1e40af")
    ax.text(1.02, 1.74, "interpolation\nthreshold", fontsize=10, color="#991b1b")
    ax.text(2.2, 0.98, "overparameterized", fontsize=10, color="#166534")
    ax.set_xlabel("model size / sample size")
    ax.set_ylabel("test risk, schematic")
    ax.set_title("Double descent is a regime-dependent risk pattern, not a universal law", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.24)
    _save(fig, "interpolation_double_descent_regimes.png")


def minimum_norm_interpolator_solution_space() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.22)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    t = np.linspace(-3, 3, 100)
    line = np.c_[t, 0.55 * t + 1.1]
    ax.plot(line[:, 0], line[:, 1], color="#64748b", lw=2.5, label="X theta = y")
    closest = np.array([-0.48, 0.84])
    ax.scatter([closest[0]], [closest[1]], s=90, color="#dc2626", zorder=4, label="min norm")
    ax.plot([0, closest[0]], [0, closest[1]], color="#dc2626", lw=2)
    ax.scatter([0], [0], s=65, color="#111827")
    ax.text(0.12, -0.18, "origin", fontsize=9)
    ax.text(closest[0] - 1.2, closest[1] + 0.18, "theta_min", fontsize=10, color="#991b1b")
    ax.text(0.65, 1.9, "infinitely many interpolators", fontsize=10, color="#334155")
    ax.set_xlabel("theta_1")
    ax.set_ylabel("theta_2")
    ax.set_title("Minimum-norm interpolation selects one point in the solution affine set", fontsize=14, weight="bold")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "minimum_norm_interpolator_solution_space.png")


def implicit_bias_max_margin_trajectory() -> None:
    t = np.linspace(1, 120, 240)
    norm = np.log(t) + 0.25 * np.sqrt(t / 120)
    angle = 42 * np.exp(-t / 36) + 3
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    axes[0].plot(t, norm, color="#dc2626", lw=2.4)
    axes[0].set_title("parameter norm grows", fontsize=12, weight="bold")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("||w_t||")
    axes[0].grid(True, alpha=0.24)
    axes[1].plot(t, angle, color="#2563eb", lw=2.4)
    axes[1].set_title("direction approaches max margin", fontsize=12, weight="bold")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("angle to SVM direction")
    axes[1].grid(True, alpha=0.24)
    fig.suptitle("Separable logistic regression: norm diverges while direction converges", fontsize=14, weight="bold")
    _save(fig, "implicit_bias_max_margin_trajectory.png")


def ntk_linearization_tangent_features() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    x = np.linspace(-2.3, 2.3, 300)
    y = np.tanh(1.2 * x) + 0.18 * x**2
    x0 = 0.35
    y0 = np.tanh(1.2 * x0) + 0.18 * x0**2
    slope = 1.2 * (1 - np.tanh(1.2 * x0) ** 2) + 0.36 * x0
    axes[0].plot(x, y, color="#111827", lw=2.4, label="f_theta")
    axes[0].plot(x, y0 + slope * (x - x0), color="#dc2626", lw=2.0, linestyle="--", label="local tangent")
    axes[0].scatter([x0], [y0], color="#dc2626", s=70)
    axes[0].set_title("local linearization", fontsize=12, weight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.22)

    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 5)
    _box(ax, (0.6, 3.5), 1.6, 0.75, "input x", "#dbeafe")
    _box(ax, (2.7, 3.5), 1.8, 0.75, "Jacobian\nfeatures J", "#dcfce7")
    _box(ax, (2.55, 1.6), 2.1, 0.85, "K = J J^T", "#fef3c7")
    _box(ax, (5.0, 3.5), 1.0, 0.75, "output", "#fee2e2")
    _arrow(ax, (2.2, 3.88), (2.7, 3.88))
    _arrow(ax, (4.5, 3.88), (5.0, 3.88))
    _arrow(ax, (3.6, 3.5), (3.6, 2.45))
    ax.text(3.6, 0.78, "NTK is a kernel built from tangent features", ha="center", fontsize=10, color="#334155")
    fig.suptitle("NTK starts from first-order network linearization around initialization", fontsize=14, weight="bold")
    _save(fig, "ntk_linearization_tangent_features.png")


def lazy_training_vs_feature_learning() -> None:
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(24, 2))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharex=True, sharey=True)
    for ax, title in zip(axes, ["lazy / kernel-like", "feature learning"]):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.22)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(title, fontsize=12, weight="bold")
    axes[0].scatter(pts[:, 0], pts[:, 1], color="#94a3b8", s=35)
    axes[0].scatter(pts[:, 0] + 0.06, pts[:, 1] - 0.04, color="#2563eb", s=22)
    axes[0].text(-2.35, 2.1, "geometry nearly fixed", fontsize=9, color="#334155")
    moved = pts @ np.array([[1.25, 0.45], [-0.25, 0.85]])
    axes[1].scatter(pts[:, 0], pts[:, 1], color="#cbd5e1", s=25, label="initial")
    axes[1].scatter(moved[:, 0], moved[:, 1], color="#dc2626", s=32, label="trained")
    for p, q in zip(pts[:8], moved[:8]):
        axes[1].plot([p[0], q[0]], [p[1], q[1]], color="#f59e0b", lw=1.1, alpha=0.8)
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].text(-2.35, 2.1, "representation changes", fontsize=9, color="#334155")
    fig.suptitle("Lazy training fixes tangent geometry; feature learning changes representation", fontsize=14, weight="bold")
    _save(fig, "lazy_training_vs_feature_learning.png")


def domain_adaptation_bound_components() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.2)
    _box(ax, (0.75, 2.3), 2.0, 1.0, "source risk\nR_S(h)", "#dbeafe")
    _box(ax, (3.25, 2.3), 2.0, 1.0, "domain\n discrepancy", "#fef3c7")
    _box(ax, (5.75, 2.3), 2.0, 1.0, "joint error\nlambda", "#fee2e2")
    _box(ax, (9.0, 2.3), 2.2, 1.0, "target risk\nR_T(h)", "#dcfce7")
    ax.text(2.98, 2.8, "+", fontsize=22, weight="bold", ha="center", va="center")
    ax.text(5.48, 2.8, "+", fontsize=22, weight="bold", ha="center", va="center")
    ax.text(8.25, 2.8, "=>", fontsize=18, weight="bold", ha="center", va="center")
    ax.text(6.1, 1.15, "small discrepancy is not enough if lambda is large", ha="center", fontsize=10, color="#334155")
    ax.set_title("Domain adaptation bounds need source risk, domain discrepancy, and joint feasibility", fontsize=14, weight="bold")
    _save(fig, "domain_adaptation_bound_components.png")


def invariant_representation_failure_case() -> None:
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    source_pos = rng.normal([-1.0, 0.7], [0.25, 0.25], size=(24, 2))
    source_neg = rng.normal([-1.0, -0.7], [0.25, 0.25], size=(24, 2))
    target_pos = rng.normal([1.0, -0.7], [0.25, 0.25], size=(24, 2))
    target_neg = rng.normal([1.0, 0.7], [0.25, 0.25], size=(24, 2))
    axes[0].scatter(source_pos[:, 0], source_pos[:, 1], color="#2563eb", marker="o", label="source +")
    axes[0].scatter(source_neg[:, 0], source_neg[:, 1], color="#dc2626", marker="o", label="source -")
    axes[0].scatter(target_pos[:, 0], target_pos[:, 1], color="#2563eb", marker="x", label="target +")
    axes[0].scatter(target_neg[:, 0], target_neg[:, 1], color="#dc2626", marker="x", label="target -")
    axes[0].set_title("before alignment", fontsize=12, weight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    axes[0].grid(True, alpha=0.22)

    z_pos = rng.normal([0.0, 0.15], [0.2, 0.16], size=(48, 2))
    z_neg = rng.normal([0.0, -0.15], [0.2, 0.16], size=(48, 2))
    axes[1].scatter(z_pos[:, 0], z_pos[:, 1], color="#2563eb", s=32)
    axes[1].scatter(z_neg[:, 0], z_neg[:, 1], color="#dc2626", s=32)
    axes[1].axhspan(-0.05, 0.05, color="#fef3c7", alpha=0.7)
    axes[1].text(-0.55, 0.48, "marginals aligned,\nlabel structure weakened", fontsize=9, color="#334155")
    axes[1].set_title("after forced invariance", fontsize=12, weight="bold")
    axes[1].grid(True, alpha=0.22)
    for ax in axes:
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.45, 1.45)
    fig.suptitle("Domain-invariant representation need not preserve predictive mechanism", fontsize=14, weight="bold")
    _save(fig, "invariant_representation_failure_case.png")


def modern_generalization_lenses_map() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 8.0))
    ax.set_axis_off()
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-3.6, 3.6)
    ax.add_patch(Circle((0, 0), 0.95, facecolor="#fef3c7", edgecolor="#92400e", lw=1.5))
    ax.text(0, 0, "selected\npredictor", ha="center", va="center", fontsize=11, weight="bold")
    labels = [
        ("VC / uniform", 0),
        ("Rademacher", 45),
        ("stability", 90),
        ("margin / norm", 135),
        ("implicit bias", 180),
        ("benign\ninterpolation", 225),
        ("NTK", 270),
        ("domain\nadaptation", 315),
    ]
    colors = ["#dbeafe", "#e0f2fe", "#dcfce7", "#fef3c7", "#ede9fe", "#fee2e2", "#fce7f3", "#ccfbf1"]
    for (label, angle), color in zip(labels, colors):
        rad = np.deg2rad(angle)
        x, y = 3.25 * np.cos(rad), 2.35 * np.sin(rad)
        _box(ax, (x - 0.75, y - 0.32), 1.5, 0.64, label, color, fontsize=9)
        _arrow(ax, (0.9 * np.cos(rad), 0.65 * np.sin(rad)), (x - 0.78 * np.cos(rad), y - 0.34 * np.sin(rad)), color="#64748b", lw=1.2)
    ax.set_title("Modern generalization theory uses multiple scoped lenses", fontsize=14, weight="bold")
    _save(fig, "modern_generalization_lenses_map.png")


def theorem_to_real_system_extrapolation() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.4)
    stages = [
        ("phenomenon", "#dbeafe"),
        ("surrogate\nmodel", "#e0f2fe"),
        ("theorem", "#dcfce7"),
        ("interpretation", "#fef3c7"),
        ("real system\nclaim", "#fee2e2"),
    ]
    xs = [0.55, 3.0, 5.45, 7.9, 10.55]
    widths = [1.75, 1.75, 1.55, 1.85, 1.85]
    for (label, color), x0, width in zip(stages, xs, widths):
        _box(ax, (x0, 2.65), width, 0.9, label, color)
    for x0, width, nxt in zip(xs[:-1], widths[:-1], xs[1:]):
        _arrow(ax, (x0 + width, 3.1), (nxt, 3.1))
    _box(ax, (4.05, 0.95), 2.2, 0.78, "assumptions", "#ede9fe")
    _box(ax, (8.85, 0.95), 2.2, 0.78, "bridge needed", "#fce7f3")
    _arrow(ax, (5.15, 1.73), (6.2, 2.65), "#7c3aed")
    _arrow(ax, (9.95, 1.73), (11.1, 2.65), "#be185d")
    ax.text(6.6, 4.25, "Do not hide extrapolation inside theorem language", ha="center", fontsize=11, color="#334155")
    ax.set_title("Audit the path from theorem to claim about a real ML system", fontsize=14, weight="bold")
    _save(fig, "theorem_to_real_system_extrapolation.png")


def main() -> None:
    class_vs_algorithm_dependent_generalization()
    algorithmic_stability_neighboring_datasets()
    rademacher_random_sign_complexity()
    norm_bounded_rademacher_geometry()
    interpolation_double_descent_regimes()
    minimum_norm_interpolator_solution_space()
    implicit_bias_max_margin_trajectory()
    ntk_linearization_tangent_features()
    lazy_training_vs_feature_learning()
    domain_adaptation_bound_components()
    invariant_representation_failure_case()
    modern_generalization_lenses_map()
    theorem_to_real_system_extrapolation()


if __name__ == "__main__":
    main()
