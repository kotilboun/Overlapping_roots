#!/usr/bin/env python3
"""Step 2/2 for Figure 4: draw the figure from the cached canonical QSS tables.

Reads ``data/perturbation_data.npz`` and ``data/configuration.json`` (written
by ``01_generate_data.py``) and draws the five-panel perturbation-scale and
accuracy figure. The PDF is written directly to the manuscript's
``figures/Fig4.pdf`` -- the exact path ``\\includegraphics{figures/Fig4.pdf}``
in sn-article.tex uses -- so re-running this script is the only step needed to
refresh the manuscript figure. It never re-solves the canonical QSS branch.

Run (after 01_generate_data.py has populated data/):
    python 02_make_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
# Canonical location LaTeX includes; this script is the only writer of Fig4.pdf.
MANUSCRIPT_FIGURES_DIR = HERE.parents[1] / "figures"

CURVE_R_VALUES = (1.0, 1.5, 2.0, 4.0)
PANEL_A_R_MAX = 10.0
PANEL_E_R_MAX = 10.0
R_COLORS = {1.0: "#C44E52", 1.5: "#6C55A3", 2.0: "#2878B5", 4.0: "#3C8D55"}
PANEL_E_COLORS = {"canonical_qss": "#1A1A1A", "p0": "#808080", "p1": "#008C95"}
MARKERS = ("o", "s", "^", "D")
LINESTYLES = ("-", "-", "-", "-")


def epsilon(R: float | np.ndarray, c: float | np.ndarray) -> float | np.ndarray:
    return c * R / (R + 1.0) ** 2


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_heading(ax: plt.Axes, letter: str, text: str) -> None:
    ax.text(0.0, 1.045, rf"$\mathbf{{({letter})}}$ {text}", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)


def make_figure(arrays: dict[str, np.ndarray], dpi: int) -> plt.Figure:
    set_style()
    colors = [R_COLORS[R] for R in CURVE_R_VALUES]
    fig = plt.figure(figsize=(6.85, 7.45), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=(0.78, 1.12, 0.12, 1.44))

    scale_ax = fig.add_subplot(grid[0, :])
    R_mesh = np.linspace(1.0e-3, PANEL_A_R_MAX, 501)
    c_mesh = np.linspace(0.0, 1.0, 241)
    RR, CC = np.meshgrid(R_mesh, c_mesh)
    epsilon_mesh = np.asarray(epsilon(RR, CC))
    image = scale_ax.pcolormesh(RR, CC, epsilon_mesh, shading="auto", cmap="viridis", vmin=0.0, vmax=0.25, rasterized=True)
    contours = scale_ax.contour(RR, CC, epsilon_mesh, levels=(0.05, 0.10, 0.15, 0.20), colors="white", linewidths=0.65)
    scale_ax.clabel(contours, fmt="%.2f", fontsize=10)
    scale_ax.plot(1.0, 1.0, marker="o", markersize=5.0, markerfacecolor="none", markeredgecolor="white", markeredgewidth=1.0)
    scale_ax.text(0.52, 0.88, r"$\varepsilon_{\max}=1/4$", color="white", fontsize=10)
    scale_ax.text(0.01, 0.03, r"left boundary: $R\to0^+$", transform=scale_ax.transAxes, color="white", fontsize=10)
    colorbar = fig.colorbar(image, ax=scale_ax, pad=0.012, fraction=0.045)
    colorbar.set_label(r"$\varepsilon=cR/(R+1)^2$")
    colorbar.set_ticks((0.00, 0.05, 0.10, 0.15, 0.20, 0.25))
    scale_ax.set_xlim(0.0, PANEL_A_R_MAX)
    scale_ax.set_ylim(0.0, 1.0)
    scale_ax.set_xlabel(r"$R$")
    scale_ax.set_ylabel(r"$c$")
    scale_ax.set_xticks((0, 1, 2, 4, 6, 8, 10))
    panel_heading(scale_ax, "a", r"Perturbation scale $\varepsilon(R,c)$")

    c_grid = arrays["c_grid"]
    error_axes: list[plt.Axes] = []
    for order in (0, 1, 2):
        ax = fig.add_subplot(grid[1, order], sharey=error_axes[0] if error_axes else None)
        error_axes.append(ax)
        if order > 0:
            ax.tick_params(labelleft=False)
        for R_index, R in enumerate(CURVE_R_VALUES):
            error = arrays["absolute_error"][R_index, order]
            ax.plot(
                c_grid,
                np.where(error > 0.0, error, np.nan),
                color=colors[R_index],
                linestyle=LINESTYLES[R_index],
                marker=MARKERS[R_index],
                markevery=max(1, (len(c_grid) - 1) // 5),
                linewidth=1.1,
                markersize=3.5,
                markeredgewidth=0.7,
            )
        ax.set_yscale("log")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0e-13, 3.0e-1)
        ax.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
        ax.set_xticklabels(("0", "0.25", "0.5", "0.75", "1"))
        ax.set_xlabel(r"$c$")
        ax.grid(which="major", color="0.86", linewidth=0.6)
        ax.grid(which="minor", axis="y", color="0.93", linewidth=0.35)
        panel_heading(ax, chr(98 + order), rf"retained order $p={order}$")
        if order == 2:
            R2_index = CURVE_R_VALUES.index(2.0)
            R2_signed_error = arrays["signed_error"][R2_index, order]
            sign_changes = np.where(
                (R2_signed_error[:-1] != 0.0) & (R2_signed_error[1:] != 0.0) & (np.signbit(R2_signed_error[:-1]) != np.signbit(R2_signed_error[1:]))
            )[0]
            positive_c_changes = [index for index in sign_changes if c_grid[index] > 0.0]
            if positive_c_changes:
                crossing_index = positive_c_changes[0]
                local_indices = (crossing_index, crossing_index + 1)
                dip_index = min(local_indices, key=lambda index: abs(R2_signed_error[index]))
                ax.annotate(
                    r"$m_1^{[2]}-m_1^{\mathrm{QSS}}=0$",
                    xy=(float(c_grid[dip_index]), float(abs(R2_signed_error[dip_index]))),
                    xytext=(0.48, 3.0e-9),
                    color=R_COLORS[2.0],
                    fontsize=10.0,
                    ha="center",
                    va="center",
                    arrowprops={"arrowstyle": "->", "color": R_COLORS[2.0], "linewidth": 0.75},
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
                )
    error_axes[0].set_ylabel(r"$e_p^{\mathrm{pert}}=|m_1^{[p]}-m_1^{\mathrm{QSS}}|$")

    legend_ax = fig.add_subplot(grid[2, :])
    legend_ax.axis("off")

    bottom_grid = grid[3, :].subgridspec(1, 5)
    m1_ax = fig.add_subplot(bottom_grid[0, 1:4])
    R_grid = arrays["m1_vs_R_grid"]
    m1_ax.plot(R_grid, arrays["m1_vs_R_canonical"], color=PANEL_E_COLORS["canonical_qss"], linewidth=1.35, label="selected QSS branch")
    m1_ax.plot(R_grid, arrays["m1_vs_R_zeroth_order"], color=PANEL_E_COLORS["p0"], linewidth=1.15, linestyle=":", label=r"$p=0$")
    m1_ax.plot(R_grid, arrays["m1_vs_R_first_order"], color=PANEL_E_COLORS["p1"], linewidth=1.15, linestyle="-.", label=r"$p=1$")
    m1_ax.set_xlim(0.0, PANEL_E_R_MAX)
    m1_ax.set_ylim(0.0, 1.0)
    m1_ax.set_xticks((0, 1, 2, 4, 6, 8, 10))
    m1_ax.set_yticks((0.0, 0.5, 1.0))
    m1_ax.set_xlabel(r"$R$")
    m1_ax.set_ylabel(r"$m_1$")
    m1_ax.grid(color="0.86", linewidth=0.6)
    panel_heading(m1_ax, "e", r"$m_1(R,c=1)$")
    m1_ax.legend(frameon=False, loc="lower center", ncol=3, handlelength=1.8, columnspacing=0.9, handletextpad=0.45)

    legend_handles = [
        Line2D([0], [0], color=colors[index], linestyle=LINESTYLES[index], marker=MARKERS[index], linewidth=1.1, markersize=3.5, label=rf"$R={R:g}$")
        for index, R in enumerate(CURVE_R_VALUES)
    ]
    legend_ax.legend(handles=legend_handles, loc="center", ncol=4, frameon=False, handlelength=1.8, columnspacing=1.0, handletextpad=0.4)

    return fig


def caption_text(configuration_sha256: str, arrays: dict[str, np.ndarray]) -> str:
    maximum_depth = int(np.max(arrays["accepted_depth"]))
    maximum_residual = float(np.max(arrays["fixed_point_residual"]))
    maximum_depth_difference = float(np.max(arrays["depth_doubling_difference"]))
    maximum_bessel_difference = float(np.max(arrays["bessel_cf_difference"]))
    return (
        "Fig. 4 Perturbation scale and approximation error relative to the canonical "
        "selected QSS branch. (a) The exact scale epsilon=cR/(R+1)^2 over "
        "0<R<=10 and 0<=c<=1; epsilon<=1/4, with equality at R=1,c=1. "
        "The displayed left boundary is the limit R->0+ and is not treated as a "
        "numerical point in the R>0 theorem. (b-d) Absolute empirical errors "
        "e_p^pert=|m1^[p]-m1^QSS| for retained orders p=0,1,2 over "
        "R in {1,1.5,2,4}. The coefficient recursion has formal finite-depth "
        "residual orders O(epsilon^(p+1)); these plotted errors instead use the "
        "canonical selected infinite-tail branch and do not assert a uniform "
        "infinite-hierarchy error theorem. Exactly zero errors at c=0 are omitted "
        "from logarithmic axes. The narrow minimum in the R=2 curve in panel (d) "
        "occurs where the signed error m1^[2]-m1^QSS changes sign and therefore "
        "does not indicate numerical instability; its crossing bracket is listed "
        "in data/perturbation_error_summary.csv. (e) Along c=1, the canonical "
        "QSS branch is compared with the p=0 and p=1 approximations for "
        "0.02<=R<=10. The first-order approximation is already nearly "
        "indistinguishable from the canonical branch over the displayed range. "
        "The left boundary of panels (a) and (e) denotes the limit R->0+ rather "
        "than a numerical point. "
        f"The adaptive zero-terminal continued fraction used depth at most {maximum_depth}, "
        f"with maximum fixed-point residual {maximum_residual:.3e}, maximum accepted "
        f"depth-doubling difference {maximum_depth_difference:.3e}, and maximum "
        f"scaled-Bessel/continued-fraction difference {maximum_bessel_difference:.3e}. "
        f"Analytical table configuration SHA-256: {configuration_sha256}."
    )


def main() -> None:
    if not (DATA_DIR / "perturbation_data.npz").exists():
        raise FileNotFoundError(f"{DATA_DIR} is missing cached data; run 01_generate_data.py first.")

    config = json.loads((DATA_DIR / "configuration.json").read_text(encoding="utf-8"))
    with np.load(DATA_DIR / "perturbation_data.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}

    fig = make_figure(arrays, dpi=600)

    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = MANUSCRIPT_FIGURES_DIR / "Fig4.pdf"
    fig.savefig(pdf_path)
    fig.savefig(HERE / "Fig4_preview.png", dpi=200)
    plt.close(fig)

    (HERE / "Fig4_caption.txt").write_text(
        caption_text(config["configuration_sha256"], arrays) + "\n", encoding="utf-8"
    )
    print(f"Wrote {pdf_path} (LaTeX-linked) and {HERE / 'Fig4_preview.png'}.")


if __name__ == "__main__":
    main()
