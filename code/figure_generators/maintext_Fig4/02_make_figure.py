#!/usr/bin/env python3
"""Step 2/2 for Figure 7: draw the figure from the cached trajectory-error table.

Reads only ``data/trajectory_error_summary.csv`` (written by
``01_generate_data.py``) and draws the four-panel log-scale trajectory-error
figure. The PDF is written directly to the manuscript's ``figures/Fig7.pdf``
-- the exact path ``\\includegraphics{figures/Fig7.pdf}`` in manuscript.tex
uses -- so re-running this script is the only step needed to refresh what
LaTeX renders. It never re-simulates the ABM or re-solves the closures.

Run (after 01_generate_data.py has populated data/):
    python 02_make_figure.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
# Canonical location LaTeX includes; this script is the only writer of Fig7.pdf.
MANUSCRIPT_FIGURES_DIR = HERE.parents[1] / "figures"

R_VALUES = (1.0, 1.5, 2.0, 4.0)
C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
CLOSURES = ("algebraic_qss0", "dynamic_K1", "dynamic_K2", "dynamic_K3")
PANEL_TITLES = (
    r"Algebraic QSS, $E_{\mathrm{QSS}}^{(0)}$",
    r"Dynamic $K=1$, $E_{1}^{\mathrm{dyn}}$",
    r"Dynamic $K=2$, $E_{2}^{\mathrm{dyn}}$",
    r"Dynamic $K=3$, $E_{3}^{\mathrm{dyn}}$",
)
PANEL_LETTERS = ("a", "b", "c", "d")
R_STYLES = {
    1.0: {"color": "#C9585A", "marker": "o", "linestyle": "-", "label": r"$R=1$"},
    1.5: {"color": "#7356A5", "marker": "s", "linestyle": "-", "label": r"$R=1.5$"},
    2.0: {"color": "#1F77B4", "marker": "^", "linestyle": "-", "label": r"$R=2$"},
    4.0: {"color": "#348A4B", "marker": "D", "linestyle": "-", "label": r"$R=4$"},
}
LOG_FLOOR = 1.0e-4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_figure(rows: list[dict[str, str]]) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.55), sharex=True, sharey=True)
    handles = []
    for panel, (ax, closure, title, letter) in enumerate(zip(axes.flat, CLOSURES, PANEL_TITLES, PANEL_LETTERS)):
        for R in R_VALUES:
            selected = [row for row in rows if row["closure"] == closure and float(row["R"]) == R]
            selected.sort(key=lambda row: float(row["c"]))
            x = np.asarray([row["c"] for row in selected], dtype=float)
            y = np.asarray([row["error"] for row in selected], dtype=float)
            low = np.asarray([row["ci_low"] for row in selected], dtype=float)
            high = np.asarray([row["ci_high"] for row in selected], dtype=float)
            style = R_STYLES[R]
            low_plot = np.maximum(low, LOG_FLOOR)
            ax.vlines(x, low_plot, high, color=style["color"], linewidth=1.0, zorder=2)
            ax.hlines(high, x - 0.012, x + 0.012, color=style["color"], linewidth=1.0, zorder=2)
            positive_low = low > 0.0
            ax.hlines(
                low[positive_low], x[positive_low] - 0.012, x[positive_low] + 0.012,
                color=style["color"], linewidth=1.0, zorder=2,
            )
            reaches_zero = ~positive_low
            if np.any(reaches_zero):
                ax.scatter(
                    x[reaches_zero], np.full(np.count_nonzero(reaches_zero), LOG_FLOOR * 1.06),
                    color=style["color"], marker="v", s=20, linewidths=0.0, zorder=4, clip_on=False,
                )
            (artist,) = ax.plot(
                x, y, color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                linewidth=1.8, markersize=5.8, markerfacecolor=style["color"], markeredgecolor=style["color"],
                label=style["label"], zorder=3,
            )
            if panel == 0:
                handles.append(artist)
        ax.set_title(title, pad=7)
        ax.text(
            0.96, 0.96, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right",
        )
        ax.set_yscale("log")
        ax.set_ylim(LOG_FLOOR, 5.0e-2)
        ax.set_xticks(C_VALUES)
        ax.set_xlim(-0.035, 1.035)
        ax.grid(axis="y", which="major", color="#D7D7D7", linewidth=0.65, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supylabel(r"Trajectory error, $E_C$", x=0.018)
    fig.supxlabel(r"Tracing intensity, $c$", y=0.025)

    fig.legend(
        handles=handles, labels=[R_STYLES[R]["label"] for R in R_VALUES],
        loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False,
        handlelength=2.7, columnspacing=1.7, handletextpad=0.55,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.72, wspace=0.12)
    return fig


def caption_text() -> str:
    return (
        "Fig. 7 Closure-trajectory validation against the ensemble-mean finite-population "
        "agent-based model (ABM). The normalized integrated L2 trajectory error E_C in "
        "Eq. (7.16) is shown against tracing intensity c for (a) the zeroth-order algebraic "
        "QSS closure and the dynamic tail closures of depth (b) K=1, (c) K=2, and (d) K=3. "
        "The common vertical axis is logarithmic. "
        "The compared trajectory vector is z=(S/S_0,I/S_0,M_1/S_0), with M_1=I m_1 "
        "for each deterministic closure and M_1 averaged directly from the raw ABM counts. "
        "Points compare each deterministic closure with the sample ensemble-mean ABM "
        "trajectory. Error bars are 95% Monte Carlo intervals obtained by applying a "
        "first-order delta method to complete-realization influence values; they quantify "
        "uncertainty in the estimated ensemble mean rather than replicate-to-replicate "
        "trajectory spread. Downward arrowheads at the lower plotting boundary indicate "
        "intervals that reach zero, which cannot be displayed on the common logarithmic axis. "
        "Colors and markers denote R=1 (coral circles), R=1.5 (purple squares), "
        "R=2 (blue triangles), and R=4 (green diamonds); all curves are solid. Simulations use "
        "S_0=8000, I_0=160, c in {0,0.25,0.5,0.75,1}, 120 independent realizations per (R,c) "
        "cell, and 151 observation times on 0<=tau<=5; tracing is activated at tau=0.5. "
        "The integral includes the preactivation interval 0<=tau<0.5 and therefore includes "
        "initial genealogical adjustment, particularly for the algebraic QSS closure. "
        "Smaller values indicate closer agreement with the sampled ensemble-mean event-level "
        "process; most dynamic-closure differences are not resolved from Monte Carlo "
        "uncertainty at 120 realizations.\n"
    )


def main() -> None:
    if not (DATA_DIR / "trajectory_error_summary.csv").exists():
        raise FileNotFoundError(f"{DATA_DIR} is missing cached data; run 01_generate_data.py first.")

    rows = read_csv(DATA_DIR / "trajectory_error_summary.csv")
    fig = make_figure(rows)

    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = MANUSCRIPT_FIGURES_DIR / "Fig7.pdf"
    fig.savefig(
        pdf_path,
        metadata={
            "Title": "Figure 7: closure trajectory validation against ensemble-mean ABM",
            "Creator": Path(__file__).name,
        },
    )
    fig.savefig(HERE / "Fig7_preview.png", dpi=200)
    plt.close(fig)

    (HERE / "Fig7_caption.txt").write_text(caption_text(), encoding="utf-8")
    print(f"Wrote {pdf_path} (LaTeX-linked) and {HERE / 'Fig7_preview.png'}.")


if __name__ == "__main__":
    main()
