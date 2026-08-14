"""Generate original T4 Learning From Data figures."""

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


def svm_functional_vs_geometric_margin() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharex=True, sharey=True)
    x = np.linspace(-2.8, 2.8, 200)
    y_line = -0.45 * x + 0.25
    point = np.array([1.45, 1.65])

    for ax, scale in zip(axes, [1, 3]):
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.2, 3)
        ax.grid(True, alpha=0.25)
        ax.plot(x, y_line, color="#111827", lw=2.2, label="same boundary")
        ax.scatter([point[0]], [point[1]], s=85, color="#2563eb", zorder=4)
        ax.text(point[0] + 0.12, point[1] + 0.1, "x_i, y_i=+1", fontsize=9, color="#1d4ed8")

        w = np.array([0.45, 1.0])
        w = w / np.linalg.norm(w)
        foot = point - ((point[1] + 0.45 * point[0] - 0.25) / np.sqrt(1 + 0.45**2)) * w
        ax.plot([point[0], foot[0]], [point[1], foot[1]], color="#dc2626", lw=2)
        ax.text((point[0] + foot[0]) / 2 + 0.1, (point[1] + foot[1]) / 2, "geometric\nmargin", fontsize=9, color="#991b1b")

        ax.arrow(-1.55, 0.95, 0.45, 1.0, width=0.025, head_width=0.15, color="#16a34a", length_includes_head=True)
        ax.set_title(f"parameters scaled by c={scale}", fontsize=12, weight="bold")
        ax.text(-2.7, -1.75, f"functional margin scales by {scale}x\nboundary and distance do not", fontsize=9, color="#334155")
        ax.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig.suptitle("Functional margin changes under parameter scaling; geometric margin does not", fontsize=14, weight="bold")
    _save(fig, "svm_functional_vs_geometric_margin.png")


def svm_maximum_margin_support_vectors() -> None:
    rng = np.random.default_rng(14)
    pos = rng.normal([1.2, 1.0], [0.35, 0.35], size=(18, 2))
    neg = rng.normal([-1.1, -0.75], [0.35, 0.35], size=(18, 2))
    support = np.array([[0.1, 0.95], [0.65, 0.3], [-0.35, -0.2]])
    labels = np.array([1, 1, -1])

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.22)
    ax.scatter(pos[:, 0], pos[:, 1], color="#2563eb", s=45, label="+1")
    ax.scatter(neg[:, 0], neg[:, 1], color="#dc2626", s=45, label="-1")
    for p, lab in zip(support, labels):
        color = "#2563eb" if lab == 1 else "#dc2626"
        ax.scatter([p[0]], [p[1]], color=color, s=65, edgecolor="#111827", zorder=5)
        ax.add_patch(Circle((p[0], p[1]), 0.16, fill=False, edgecolor="#f59e0b", lw=2.2))

    x = np.linspace(-2.2, 2.2, 200)
    boundary = -0.45 * x + 0.25
    ax.plot(x, boundary, color="#111827", lw=2.2, label="decision boundary")
    ax.plot(x, boundary + 0.62, color="#64748b", lw=1.6, linestyle="--", label="margin planes")
    ax.plot(x, boundary - 0.62, color="#64748b", lw=1.6, linestyle="--")
    ax.fill_between(x, boundary - 0.62, boundary + 0.62, color="#fef3c7", alpha=0.45)
    ax.text(0.9, -0.55, "maximum empty band\nunder chosen representation", fontsize=10, color="#92400e")
    ax.set_xlim(-2.3, 2.4)
    ax.set_ylim(-2.0, 2.2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Maximum-margin separator is determined by active constraints", fontsize=14, weight="bold")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "svm_maximum_margin_support_vectors.png")


def svm_primal_dual_structure() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.4)

    _box(ax, (0.6, 3.6), 2.35, 0.9, "primal variables\nw, b", "#dbeafe")
    _box(ax, (0.6, 2.1), 2.35, 0.9, "constraints\ny_i(w^T x_i+b) >= 1", "#e0e7ff", fontsize=9)
    _box(ax, (3.9, 2.85), 2.4, 0.9, "Lagrangian\nalpha_i >= 0", "#fef3c7")
    _box(ax, (7.05, 3.6), 2.35, 0.9, "stationarity\nw=sum alpha_i y_i x_i", "#dcfce7", fontsize=9)
    _box(ax, (7.05, 2.1), 2.35, 0.9, "dual variables\nalpha_i", "#ede9fe")
    _box(ax, (10.25, 2.85), 1.9, 0.9, "classifier\nsum over SVs", "#fee2e2")

    _arrow(ax, (2.95, 4.05), (3.9, 3.55))
    _arrow(ax, (2.95, 2.55), (3.9, 3.1))
    _arrow(ax, (6.3, 3.3), (7.05, 4.05))
    _arrow(ax, (6.3, 3.3), (7.05, 2.55))
    _arrow(ax, (9.4, 4.05), (10.25, 3.35))
    _arrow(ax, (9.4, 2.55), (10.25, 3.15))
    ax.text(6.4, 1.0, "KKT complementary slackness zeroes non-active examples in the dual expansion", ha="center", fontsize=10, color="#334155")
    ax.set_title("SVM primal geometry becomes dual coefficient structure", fontsize=14, weight="bold")
    _save(fig, "svm_primal_dual_structure.png")


def soft_margin_slack_hinge_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))

    ax = axes[0]
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.22)
    ax.set_xlim(-2.3, 2.5)
    ax.set_ylim(-2.0, 2.2)
    x = np.linspace(-2.4, 2.4, 200)
    boundary = -0.35 * x
    ax.plot(x, boundary, color="#111827", lw=2.2)
    ax.plot(x, boundary + 0.65, color="#64748b", lw=1.4, linestyle="--")
    ax.plot(x, boundary - 0.65, color="#64748b", lw=1.4, linestyle="--")
    pts = np.array([[-1.2, 1.35], [0.0, 0.55], [1.0, 0.15], [1.5, -0.55], [-0.7, -0.2]])
    labs = [1, 1, 1, -1, -1]
    notes = ["outside margin", "on/near margin", "inside margin", "correct side", "misclassified"]
    for p, lab, note in zip(pts, labs, notes):
        color = "#2563eb" if lab == 1 else "#dc2626"
        ax.scatter([p[0]], [p[1]], color=color, s=70, edgecolor="#111827", zorder=4)
        ax.text(p[0] + 0.08, p[1] + 0.08, note, fontsize=8.5, color="#334155")
    ax.annotate("slack measures\nconstraint violation", xy=(-0.7, -0.2), xytext=(-1.75, -1.15), arrowprops=dict(arrowstyle="->", color="#dc2626"), fontsize=9, color="#991b1b")
    ax.set_title("Soft-margin geometry", fontsize=12, weight="bold")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax = axes[1]
    s = np.linspace(-1.5, 2.5, 250)
    hinge = np.maximum(0, 1 - s)
    ax.plot(s, hinge, color="#dc2626", lw=2.5)
    ax.axvline(1, color="#64748b", lw=1.4, linestyle="--")
    ax.axvline(0, color="#111827", lw=1.0, linestyle=":")
    ax.text(1.04, 0.12, "margin reached", fontsize=9, color="#334155")
    ax.text(-1.35, 2.2, "misclassified\nor severe violation", fontsize=9, color="#991b1b")
    ax.text(0.2, 0.58, "correct but\ninside margin", fontsize=9, color="#92400e")
    ax.text(1.25, 0.12, "zero hinge loss", fontsize=9, color="#166534")
    ax.set_xlabel("signed score y f(x)")
    ax.set_ylabel("max(0, 1 - y f(x))")
    ax.set_title("Hinge loss is margin-aware", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.25)

    fig.suptitle("Soft margin trades wider norm-controlled separators against violations", fontsize=14, weight="bold")
    _save(fig, "soft_margin_slack_hinge_geometry.png")


def kernel_feature_space_geometry() -> None:
    rng = np.random.default_rng(5)
    theta_inner = rng.uniform(0, 2 * np.pi, 45)
    r_inner = rng.normal(0.65, 0.08, 45)
    theta_outer = rng.uniform(0, 2 * np.pi, 55)
    r_outer = rng.normal(1.45, 0.09, 55)
    inner = np.c_[r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]
    outer = np.c_[r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    axes[0].set_aspect("equal")
    axes[0].scatter(inner[:, 0], inner[:, 1], color="#2563eb", s=35, label="class +")
    axes[0].scatter(outer[:, 0], outer[:, 1], color="#dc2626", s=35, label="class -")
    axes[0].set_title("Original input space", fontsize=12, weight="bold")
    axes[0].text(-1.8, -1.75, "nonlinear boundary in x-space", fontsize=9, color="#334155")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")

    z_inner = np.c_[inner[:, 0], inner[:, 0] ** 2 + inner[:, 1] ** 2]
    z_outer = np.c_[outer[:, 0], outer[:, 0] ** 2 + outer[:, 1] ** 2]
    axes[1].scatter(z_inner[:, 0], z_inner[:, 1], color="#2563eb", s=35)
    axes[1].scatter(z_outer[:, 0], z_outer[:, 1], color="#dc2626", s=35)
    axes[1].axhline(1.0, color="#111827", lw=2.2)
    axes[1].text(-1.75, 1.08, "linear separator in Phi-space", fontsize=9, color="#111827")
    axes[1].set_xlabel("z1")
    axes[1].set_ylabel("z2 = ||x||^2")
    axes[1].set_title("Feature-space geometry", fontsize=12, weight="bold")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("A linear separator after transformation can be nonlinear in the original input", fontsize=14, weight="bold")
    _save(fig, "kernel_feature_space_geometry.png")


def kernel_gram_psd_geometry() -> None:
    x = np.array([-1.4, -0.5, 0.25, 1.05, 1.55])
    sigma = 0.75
    gram = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * sigma**2))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    im = axes[0].imshow(gram, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Gram matrix K_ij = K(x_i, x_j)", fontsize=12, weight="bold")
    axes[0].set_xticks(range(len(x)))
    axes[0].set_yticks(range(len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            axes[0].text(j, i, f"{gram[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#111827")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)
    _box(ax, (0.5, 3.4), 2.1, 0.75, "finite weights\nc_i", "#dbeafe")
    _box(ax, (3.0, 3.4), 2.4, 0.75, "feature vectors\nPhi(x_i)", "#e0e7ff")
    _box(ax, (2.0, 1.65), 3.2, 0.9, "c^T K c = || sum_i c_i Phi(x_i) ||^2 >= 0", "#dcfce7", fontsize=9)
    _arrow(ax, (2.6, 3.78), (3.0, 3.78))
    _arrow(ax, (4.2, 3.4), (3.6, 2.55))
    _arrow(ax, (1.55, 3.4), (2.9, 2.55))
    ax.text(3.6, 0.85, "PSD is the algebraic footprint of an inner-product representation", ha="center", fontsize=10, color="#166534")
    fig.suptitle("Valid kernels produce positive semidefinite Gram matrices", fontsize=14, weight="bold")
    _save(fig, "kernel_gram_psd_geometry.png")


def rbf_local_basis_centers_widths() -> None:
    x = np.linspace(-3, 3, 400)
    centers = [-1.5, 0.0, 1.4]
    sigmas = [0.32, 0.75, 1.2]
    colors = ["#2563eb", "#16a34a", "#dc2626"]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for c, sigma, color in zip(centers, sigmas, colors):
        phi = np.exp(-((x - c) ** 2) / (2 * sigma**2))
        ax.plot(x, phi, color=color, lw=2.3, label=f"center={c}, sigma={sigma}")
        ax.axvline(c, color=color, lw=1.1, linestyle="--", alpha=0.65)
    ax.set_title("RBF basis functions encode local activation around centers", fontsize=14, weight="bold")
    ax.set_xlabel("input coordinate")
    ax.set_ylabel("basis activation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(-2.95, 0.16, "width controls spatial scale,\nnot a universal overfit law", fontsize=9, color="#334155")
    _save(fig, "rbf_local_basis_centers_widths.png")


def rbf_model_vs_kernel_svm() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    rng = np.random.default_rng(22)
    pts = rng.normal(size=(36, 2))
    labels = (pts[:, 0] + 0.45 * pts[:, 1] > 0).astype(int)
    colors = np.where(labels == 1, "#2563eb", "#dc2626")

    ax = axes[0]
    ax.set_aspect("equal")
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=35, alpha=0.75)
    centers = np.array([[-1.0, -0.7], [0.0, 0.55], [1.05, -0.1]])
    for center in centers:
        ax.scatter([center[0]], [center[1]], color="#f59e0b", s=90, marker="X", edgecolor="#111827", zorder=5)
        ax.add_patch(Circle(center, 0.75, fill=False, edgecolor="#f59e0b", lw=1.7, linestyle="--"))
    ax.set_title("RBF model: explicit finite centers", fontsize=12, weight="bold")
    ax.text(-2.1, -2.25, "basis responses are fixed once centers/widths are chosen", fontsize=9, color="#334155")
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    ax.set_aspect("equal")
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=35, alpha=0.45)
    sv_idx = [4, 8, 13, 20, 27, 31]
    ax.scatter(pts[sv_idx, 0], pts[sv_idx, 1], c=colors[sv_idx], s=80, edgecolor="#111827", zorder=5)
    for p in pts[sv_idx]:
        ax.add_patch(Circle((p[0], p[1]), 0.18, fill=False, edgecolor="#f59e0b", lw=2.0))
    ax.set_title("Kernel SVM: support-vector expansion", fontsize=12, weight="bold")
    ax.text(-2.1, -2.25, "dual coefficients select which training examples enter prediction", fontsize=9, color="#334155")
    ax.grid(True, alpha=0.22)

    fig.suptitle("RBF basis models and Gaussian-kernel SVMs use related locality ideas but different solution structures", fontsize=14, weight="bold")
    _save(fig, "rbf_model_vs_kernel_svm.png")


def three_learning_principles_failure_map() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.6)

    stages = [
        ((0.55, 3.0), "target\npopulation", "#dbeafe"),
        ((2.45, 3.0), "sampling\nmechanism", "#e0f2fe"),
        ((4.35, 3.0), "dataset", "#dcfce7"),
        ((6.25, 3.0), "candidate\nsearch", "#fef3c7"),
        ((8.15, 3.0), "selected\nmodel", "#ede9fe"),
        ((10.05, 3.0), "reported\nevidence", "#fee2e2"),
    ]
    for xy, label, color in stages:
        _box(ax, xy, 1.45, 0.8, label, color, fontsize=9)
    for xy, _, _ in stages[:-1]:
        _arrow(ax, (xy[0] + 1.45, xy[1] + 0.4), (xy[0] + 1.9, xy[1] + 0.4))

    _box(ax, (5.15, 1.1), 2.2, 0.85, "Occam\ncontrol what can be selected", "#fef3c7", fontsize=9)
    _box(ax, (1.55, 1.1), 2.3, 0.85, "Sampling bias\ncontrol what data represent", "#e0f2fe", fontsize=9)
    _box(ax, (8.65, 1.1), 2.4, 0.85, "Data snooping\ncontrol hidden feedback", "#fee2e2", fontsize=9)
    _arrow(ax, (6.25, 3.0), (6.25, 1.95), "#92400e")
    _arrow(ax, (2.7, 3.0), (2.7, 1.95), "#0369a1")
    _arrow(ax, (9.85, 3.0), (9.85, 1.95), "#991b1b")
    ax.set_title("The three learning principles act at different evidence-control points", fontsize=14, weight="bold")
    _save(fig, "three_learning_principles_failure_map.png")


def bayesian_prior_likelihood_posterior() -> None:
    theta = np.linspace(-4, 4, 500)
    prior = np.exp(-0.5 * ((theta + 0.9) / 1.1) ** 2)
    likelihood = np.exp(-0.5 * ((theta - 1.0) / 0.75) ** 2)
    posterior = prior * likelihood
    prior /= prior.max()
    likelihood /= likelihood.max()
    posterior /= posterior.max()

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.plot(theta, prior, color="#2563eb", lw=2.4, label="prior p(h)")
    ax.plot(theta, likelihood, color="#16a34a", lw=2.4, label="likelihood p(D|h)")
    ax.plot(theta, posterior, color="#dc2626", lw=2.8, label="posterior p(h|D)")
    ax.annotate("data updates assumptions", xy=(0.35, 0.98), xytext=(-2.7, 0.72), arrowprops=dict(arrowstyle="->", color="#334155"), fontsize=10, color="#334155")
    ax.set_title("Bayesian learning combines prior assumptions with data likelihood", fontsize=14, weight="bold")
    ax.set_xlabel("hypothesis coordinate, shown schematically")
    ax.set_ylabel("relative density")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(-3.95, 0.12, "the prior is an assumption;\nBayesian does not mean assumption-free", fontsize=9, color="#334155")
    _save(fig, "bayesian_prior_likelihood_posterior.png")


def aggregation_variance_correlation() -> None:
    t = np.arange(1, 61)
    rhos = [0.0, 0.25, 0.6, 0.9]
    colors = ["#16a34a", "#2563eb", "#f59e0b", "#dc2626"]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for rho, color in zip(rhos, colors):
        factor = rho + (1 - rho) / t
        ax.plot(t, factor, color=color, lw=2.3, label=f"rho={rho}")
    ax.set_title("Averaging reduces variance most when errors are weakly correlated", fontsize=14, weight="bold")
    ax.set_xlabel("number of predictors T")
    ax.set_ylabel("variance factor relative to one predictor")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(31, 0.22, "limit as T grows is rho,\nnot always zero", fontsize=10, color="#334155")
    _save(fig, "aggregation_variance_correlation.png")


def t4_geometry_representation_capacity_map() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 6.2))
    ax.set_axis_off()
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 6.2)

    stages = [
        ("World", "#dbeafe"),
        ("Observation", "#e0f2fe"),
        ("Representation\nPhi", "#dcfce7"),
        ("Geometry\ninner product / distance", "#fef3c7"),
        ("Margin /\nlocality", "#fde68a"),
        ("Objective +\nconstraint", "#ede9fe"),
        ("Selected\nsolution", "#fee2e2"),
    ]
    xs = [0.35, 2.05, 3.95, 6.0, 8.2, 10.1, 12.0]
    widths = [1.25, 1.45, 1.55, 1.7, 1.45, 1.55, 1.25]
    for (label, color), x0, width in zip(stages, xs, widths):
        _box(ax, (x0, 3.45), width, 0.9, label, color, fontsize=9)
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + widths[i], 3.9), (xs[i + 1], 3.9))

    _box(ax, (2.6, 1.35), 2.0, 0.8, "T1\nrepresentation", "#dcfce7", fontsize=9)
    _box(ax, (5.15, 1.35), 2.0, 0.8, "T2\ncapacity", "#e0e7ff", fontsize=9)
    _box(ax, (7.7, 1.35), 2.0, 0.8, "T3\nselection", "#fef3c7", fontsize=9)
    _box(ax, (10.25, 1.35), 2.0, 0.8, "T4\ngeometry + evidence", "#fee2e2", fontsize=9)
    _arrow(ax, (4.6, 1.75), (5.15, 1.75))
    _arrow(ax, (7.15, 1.75), (7.7, 1.75))
    _arrow(ax, (9.7, 1.75), (10.25, 1.75))
    ax.text(6.8, 0.55, "Ask: which arrow does a new ML paper actually modify?", ha="center", fontsize=11, color="#334155")
    ax.set_title("T4 unifies representation, induced geometry, effective complexity, and evidence discipline", fontsize=14, weight="bold")
    _save(fig, "t4_geometry_representation_capacity_map.png")


def main() -> None:
    svm_functional_vs_geometric_margin()
    svm_maximum_margin_support_vectors()
    svm_primal_dual_structure()
    soft_margin_slack_hinge_geometry()
    kernel_feature_space_geometry()
    kernel_gram_psd_geometry()
    rbf_local_basis_centers_widths()
    rbf_model_vs_kernel_svm()
    three_learning_principles_failure_map()
    bayesian_prior_likelihood_posterior()
    aggregation_variance_correlation()
    t4_geometry_representation_capacity_map()


if __name__ == "__main__":
    main()
