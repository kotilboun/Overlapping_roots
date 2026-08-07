#!/usr/bin/env python3
"""Step 2/2 for Figure 5: draw the figure from the cached analytical and ABM tables.

Reads only ``data/*.csv`` (written by ``01_generate_data.py``) and draws the
three-panel growth-rate/R_time/control-boundary figure. The PNG is written
directly to the manuscript's ``figures/Fig5.png`` -- the exact path
``\\includegraphics{figures/Fig5.png}`` in sn-article.tex uses -- so re-running
this script is the only step needed to refresh what LaTeX renders. It never
re-solves the analytical curves or re-simulates the ABM.

Run (after 01_generate_data.py has populated data/):
    python 02_make_figure.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
# Canonical location LaTeX includes; this script is the only writer of Fig5.png.
MANUSCRIPT_FIGURES_DIR = HERE.parents[1] / "figures"

R_CURVES = (1.0, 1.5, 2.0, 4.0)
R_COLORS = {1.0: "#C44E52", 1.5: "#6C55A3", 2.0: "#2878B5", 4.0: "#3C8D55"}
R_MARKERS = {1.0: "o", 1.5: "s", 2.0: "^", 4.0: "D"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{**row, **{key: float(row[key]) for key in keys}} for row in rows]


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.7,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure(
    analytical: list[dict[str, Any]],
    critical: list[dict[str, Any]],
    endpoint_R: float,
    cohort_abm: list[dict[str, Any]],
    threshold_abm: list[dict[str, Any]],
    inferred_critical_abm: list[dict[str, Any]],
) -> plt.Figure:
    publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.90))
    fig.subplots_adjust(left=0.078, right=0.985, bottom=0.29, top=0.81, wspace=0.34)

    for R in R_CURVES:
        rows = [row for row in analytical if row["R"] == R]
        c = [row["c"] for row in rows]
        axes[0].plot(c, [row["growth_rate"] for row in rows], color=R_COLORS[R], linewidth=1.35, label=rf"$R={R:g}$")
        axes[1].plot(c, [row["R_time"] for row in rows], color=R_COLORS[R], linewidth=1.35)

    for R in R_CURVES:
        growth_rows = sorted(
            (row for row in threshold_abm if row["dataset"] == "CP_COHORT_R2" and row["phase"] == "PRODUCTION" and row["R"] == R),
            key=lambda row: row["c"],
        )
        if growth_rows:
            axes[0].errorbar(
                [row["c"] for row in growth_rows],
                [row["event_growth_estimate"] for row in growth_rows],
                yerr=np.asarray(
                    [
                        [row["event_growth_estimate"] - row["bootstrap_95_low"] for row in growth_rows],
                        [row["bootstrap_95_high"] - row["event_growth_estimate"] for row in growth_rows],
                    ]
                ),
                linestyle="none",
                marker=R_MARKERS[R],
                markerfacecolor="white",
                markeredgecolor=R_COLORS[R],
                markeredgewidth=1.1,
                color=R_COLORS[R],
                markersize=5.2,
                capsize=2.3,
                elinewidth=0.9,
                zorder=5,
            )
        cohort_rows = sorted((row for row in cohort_abm if row["R"] == R), key=lambda row: row["c"])
        axes[1].errorbar(
            [row["c"] for row in cohort_rows],
            [row["pooled_R_time"] for row in cohort_rows],
            yerr=np.asarray(
                [
                    [row["pooled_R_time"] - row["bootstrap_95_low"] for row in cohort_rows],
                    [row["bootstrap_95_high"] - row["pooled_R_time"] for row in cohort_rows],
                ]
            ),
            linestyle="none",
            marker=R_MARKERS[R],
            markerfacecolor="white",
            markeredgecolor=R_COLORS[R],
            markeredgewidth=1.1,
            color=R_COLORS[R],
            markersize=5.2,
            capsize=2.3,
            elinewidth=0.9,
            zorder=5,
        )

    axes[0].axhline(0.0, color="0.25", linewidth=0.85, linestyle=":")
    axes[1].axhline(1.0, color="0.25", linewidth=0.85, linestyle=":")
    axes[0].set(xlabel=r"$c$", ylabel=r"$g$", xlim=(-0.03, 1.03))
    axes[1].set(xlabel=r"$c$", ylabel=r"$\mathcal{R}_{\mathrm{time}}$", xlim=(-0.03, 1.03))
    axes[1].legend(
        handles=[
            Line2D(
                [0], [0], linestyle="none", marker="o", markerfacecolor="white",
                markeredgecolor="#333333", markeredgewidth=1.1, markersize=5.2, label="ABM, 95% CI",
            )
        ],
        frameon=False, loc="upper right", fontsize=7.8, handletextpad=0.35, borderaxespad=0.2,
    )
    R_legend = axes[0].legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.43), ncol=4,
        handlelength=1.7, columnspacing=1.0, handletextpad=0.45,
    )
    R_legend.remove()
    fig.legend(
        handles=R_legend.legend_handles,
        labels=[text.get_text() for text in R_legend.get_texts()],
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4,
        handlelength=1.7, columnspacing=1.0, handletextpad=0.45,
    )

    axes[2].plot([row["c_star"] for row in critical], [row["R"] for row in critical], color="#111827", linewidth=1.6)
    axes[2].plot([1.0], [endpoint_R], marker="o", color="#111827", markersize=4.5, zorder=4)
    axes[2].errorbar(
        [row["c"] for row in inferred_critical_abm],
        [row["inferred_critical_R"] for row in inferred_critical_abm],
        yerr=np.asarray(
            [
                [row["inferred_critical_R"] - row["bootstrap_95_low"] for row in inferred_critical_abm],
                [row["bootstrap_95_high"] - row["inferred_critical_R"] for row in inferred_critical_abm],
            ]
        ),
        linestyle="none", marker="o", markersize=5.2, markerfacecolor="white",
        markeredgecolor="#333333", markeredgewidth=1.1, color="#333333",
        capsize=2.3, elinewidth=0.9, zorder=5, label="ABM, 95% CI",
    )
    axes[2].annotate(
        rf"$R_*(1)={endpoint_R:.6f}$",
        xy=(1.0, endpoint_R), xytext=(0.42, 1.79),
        arrowprops={"arrowstyle": "-", "color": "0.25", "lw": 0.7}, fontsize=8.5,
    )
    axes[2].set(xlabel=r"$c$", ylabel=r"critical $R$", xlim=(-0.03, 1.10), ylim=(0.97, 1.84))
    axes[2].legend(frameon=False, loc="lower right", fontsize=7.7, handletextpad=0.35, borderaxespad=0.15, labelspacing=0.2)

    titles = ("(a) Malthusian growth", "(b) incident-cohort replacement", "(c) actual control boundary")
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="left", fontweight="bold", pad=6)
        ax.grid(color="0.82", linewidth=0.55, alpha=0.55)
        ax.set_axisbelow(True)
    axes[0].set_xticks((0.0, 0.5, 1.0))
    axes[1].set_xticks((0.0, 0.5, 1.0))
    axes[2].set_xticks((0.0, 0.5, 1.0))
    axes[2].set_yticks((1.0, 1.25, 1.5, 1.75))

    return fig


def caption_text(endpoint_R: float, manifest: dict[str, Any]) -> str:
    return (
        "Fig. 5 Analytical growth, incident-cohort lifetime reproduction, and the "
        "actual tracing-control boundary under instantaneous one-step forward "
        "tracing. Panels (a) and (b) use R in {1,1.5,2,4}, with the same color "
        "mapping as Figures 3 and 4. (a) Dimensionless Malthusian rate "
        "g(R,c)=R-1-c m_1(R,c), with m_1 evaluated on the admissible QSS "
        "component continued from c=0. The horizontal line marks zero growth. "
        "Open symbols are ABM event/person-time estimates with 95% "
        "whole-replicate bootstrap intervals. (b) Incident-cohort lifetime "
        "reproduction R_time=R E[L_A]. The line R_time=1 marks lifetime "
        "replacement; the identity R_time-1=g B with B>0 makes this the same "
        "threshold contour as g=0. Open symbols are complete-follow-up ABM "
        "cohort estimates with 95% whole-replicate bootstrap intervals. "
        "(c) The same actual critical boundary plotted with c on the "
        "horizontal axis and critical R on the vertical axis; every point "
        f"satisfies g(R,c)=0. Its maximal-tracing endpoint is R_*(1)={endpoint_R:.8f}. "
        "Open circles are ABM estimates of the critical R obtained by linearly "
        "interpolating the pooled growth estimates from the two prespecified "
        "bracketing cells at each c; vertical bars are pointwise 95% "
        "whole-replicate bootstrap intervals for that interpolated critical "
        "value. ABM paths use a constant susceptible pool, many unrelated "
        "initial infectious roots, gamma/Gamma=0, gamma_c/Gamma=1, and p_f=c. "
        "CP_COHORT_R2 uses 300 independent paths per cell, completes every "
        "enrolled lifetime, and has maximum relative Monte Carlo standard "
        f"error {100.0 * manifest['maximum_relative_cohort_MCSE']:.2f}%. "
        "CP_THRESHOLD_R1 uses 300 primary paths per bracketing cell; every "
        "bracketing-cell growth interval has its prespecified sign.\n"
    )


def main() -> None:
    if not (DATA_DIR / "analytical_curves.csv").exists():
        raise FileNotFoundError(f"{DATA_DIR} is missing cached data; run 01_generate_data.py first.")

    analytical = as_float_rows(read_csv(DATA_DIR / "analytical_curves.csv"), ("R", "c", "growth_rate", "R_time"))
    critical = as_float_rows(read_csv(DATA_DIR / "critical_curve.csv"), ("R", "c_star"))
    threshold_abm = as_float_rows(
        read_csv(DATA_DIR / "abm_threshold_summary.csv"), ("R", "c", "event_growth_estimate", "bootstrap_95_low", "bootstrap_95_high")
    )
    cohort_abm = as_float_rows(
        read_csv(DATA_DIR / "abm_cohort_summary.csv"), ("R", "c", "pooled_R_time", "bootstrap_95_low", "bootstrap_95_high")
    )
    inferred_critical_abm = as_float_rows(
        read_csv(DATA_DIR / "abm_inferred_critical_points.csv"),
        ("c", "inferred_critical_R", "bootstrap_95_low", "bootstrap_95_high"),
    )
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    endpoint_R = manifest["diagnostics"]["critical_endpoint_R_at_c1"]

    fig = make_figure(analytical, critical, endpoint_R, cohort_abm, threshold_abm, inferred_critical_abm)

    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = MANUSCRIPT_FIGURES_DIR / "Fig5.png"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(HERE / "Fig5_preview.png", dpi=200, facecolor="white")
    plt.close(fig)

    (HERE / "Fig5_caption.txt").write_text(caption_text(endpoint_R, manifest), encoding="utf-8")
    print(f"Wrote {png_path} (LaTeX-linked) and {HERE / 'Fig5_preview.png'}.")


if __name__ == "__main__":
    main()
