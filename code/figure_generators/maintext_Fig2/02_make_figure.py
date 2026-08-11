#!/usr/bin/env python3
"""Step 2/2 for Figure 3: draw the figure from the cached simulation/QSS results.

Reads the CSVs and the trajectories archive written by ``01_generate_data.py``
(``data/panel_summary.csv``, ``data/qss_targets.csv``, ``data/trajectories.npz``)
and draws the six-panel ABM--QSS comparison. The PDF is written directly to the
manuscript's ``figures/Fig3.pdf`` -- the exact path
``\\includegraphics{figures/Fig3.pdf}`` in manuscript.tex uses -- so re-running
this script is the only step needed to refresh the manuscript figure. It never
re-runs simulations or root-finding.

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
# Canonical location LaTeX includes; this script is the only writer of Fig3.pdf.
MANUSCRIPT_FIGURES_DIR = HERE.parents[1] / "figures"

R_VALUES = (1.0, 1.5, 2.0, 4.0)
ORDERS = (1, 2, 3)
TAU_END = 3.25
PRIMARY_WINDOW = (3.0, 3.25)
REPRESENTATIVE_CELL = (2.0, 1.0)
REPLICATES = 120

R_COLORS = {1.0: "#C44E52", 1.5: "#6C55A3", 2.0: "#2878B5", 4.0: "#3C8D55"}
R_MARKERS = {1.0: "o", 1.5: "s", 2.0: "^", 4.0: "D"}
R_LINESTYLES = {1.0: "-", 1.5: "--", 2.0: "-.", 4.0: ":"}
ABM_COLOR = "#202020"
ABM_BAND = "#B8B8B8"
QSS_COLOR = "#D0645A"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.035,
        rf"$\mathbf{{({label})}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )


def make_figure(
    panel_rows: list[dict],
    trajectories: dict[str, np.ndarray],
    targets: dict[tuple[float, float], np.ndarray],
    dpi: int,
) -> plt.Figure:
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(6.85, 4.65))
    global_extent = max(
        abs(float(row[field])) for row in panel_rows for field in ("discrepancy_95_lower", "discrepancy_95_upper")
    )
    global_extent = max(global_extent * 1.15, 1.0e-5)
    for order_index, k in enumerate(ORDERS):
        ax = axes[0, order_index]
        for R in R_VALUES:
            rows = sorted((row for row in panel_rows if row["R"] == R and row["k"] == k), key=lambda row: row["c"])
            x = np.asarray([row["c"] for row in rows])
            y = np.asarray([row["signed_discrepancy"] for row in rows])
            low = y - np.asarray([row["discrepancy_95_lower"] for row in rows])
            high = np.asarray([row["discrepancy_95_upper"] for row in rows]) - y
            ax.errorbar(
                x,
                y,
                yerr=np.vstack((low, high)),
                color=R_COLORS[R],
                marker=R_MARKERS[R],
                linestyle=R_LINESTYLES[R],
                linewidth=1.15,
                markersize=4.0,
                capsize=2.2,
                elinewidth=0.9,
                label=rf"$R={R:g}$",
            )
        ax.axhline(0.0, color="#505050", linewidth=0.9, linestyle="-")
        ax.set_ylim(-global_extent, global_extent)
        ax.set_yticks((-0.02, 0.0, 0.02))
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.set_xticklabels(("0", "0.5", "1"))
        ax.set_xlabel(r"$c$")
        ax.set_ylabel(rf"$\widehat{{m}}_{k}^{{\rm ABM}}-m_{k}^{{\rm QSS}}$", labelpad=1)
        ax.grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, chr(97 + order_index))

    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        top_handles,
        top_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    times = trajectories["times"]
    for order_index, k in enumerate(ORDERS):
        ax = axes[1, order_index]
        mean = trajectories[f"representative_m{k}_mean"]
        low = trajectories[f"representative_m{k}_lower"]
        high = trajectories[f"representative_m{k}_upper"]
        ax.axvspan(PRIMARY_WINDOW[0], PRIMARY_WINDOW[1], color="#F3E3A1", alpha=0.55, zorder=0)
        band = ax.fill_between(
            times,
            low,
            high,
            color=ABM_BAND,
            alpha=0.55,
            linewidth=0.0,
            label="pointwise 95% confidence interval" if order_index == 0 else None,
        )
        (abm_line,) = ax.plot(
            times, mean, color=ABM_COLOR, linewidth=1.4, label="ABM mean" if order_index == 0 else None
        )
        qss_line = ax.axhline(
            targets[REPRESENTATIVE_CELL][order_index],
            color=QSS_COLOR,
            linestyle="--",
            linewidth=1.25,
            label="selected QSS target" if order_index == 0 else None,
        )
        ax.set_xlim(0.0, TAU_END)
        ax.set_ylim(bottom=0.0)
        ax.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.25))
        ax.set_xticklabels(("0", "0.5", "1", "1.5", "2", "2.5", "3.25"))
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(rf"$m_{k}(\tau)$")
        ax.grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, chr(100 + order_index))

    fig.legend(
        handles=(abm_line, band, qss_line),
        labels=("ABM mean", "pointwise 95% confidence interval", "selected QSS target"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.17, top=0.89, wspace=0.52, hspace=0.48)
    return fig


def caption_text(replicates: int) -> str:
    return (
        "Fig. 3 Constant-pool stochastic comparison with the implicit "
        "modified-Bessel QSS solution. Panels (a-c) show the signed differences "
        "between replicate-level ABM estimates and the selected QSS targets for "
        "m1, m2, and m3, respectively, over R in {1,1.5,2,4} and "
        "c in {0,0.25,0.5,0.75,1}. Points are means of replicate-level time "
        "averages over 3<=tau<=3.25; error bars are replicate-bootstrap 95% "
        "confidence intervals. Panels (d-f) show the relaxation of m1, m2, and "
        "m3 for the representative case R=2, c=1. Solid curves and "
        "shaded bands are ABM ensemble means and gray pointwise "
        "replicate-bootstrap 95% confidence intervals; horizontal dashed lines "
        "are the corresponding selected QSS targets, and the pale-yellow regions "
        "denote the averaging window 3<=tau<=3.25. "
        f"Each parameter pair used {replicates} independent realizations "
        "initialized with 160 unrelated infectious roots. Simulations used "
        "Gamma=1, gamma=0, gamma_c=1, and p_f=c, with a constant susceptible "
        "pool and tracing active from tau=0. Realizations were followed to "
        "tau=3.25; normalized coordinates were left undefined after extinction, "
        "and no terminal value was carried forward."
    )


def main() -> None:
    if not (DATA_DIR / "panel_summary.csv").exists():
        raise FileNotFoundError(f"{DATA_DIR} is missing cached data; run 01_generate_data.py first.")

    panel_rows = []
    for row in read_csv("panel_summary.csv"):
        row = {key: float(value) for key, value in row.items()}
        row["k"] = int(row["k"])
        panel_rows.append(row)

    targets: dict[tuple[float, float], np.ndarray] = {}
    for row in read_csv("qss_targets.csv"):
        key = (float(row["R"]), float(row["c"]))
        targets[key] = np.asarray([float(row[f"m{k}_qss"]) for k in ORDERS])

    with np.load(DATA_DIR / "trajectories.npz") as archive:
        trajectories = {name: archive[name] for name in archive.files}

    fig = make_figure(panel_rows, trajectories, targets, dpi=600)

    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = MANUSCRIPT_FIGURES_DIR / "Fig3.pdf"
    fig.savefig(
        pdf_path,
        metadata={"Title": "Constant-pool stochastic comparison with the selected QSS branch"},
    )
    fig.savefig(HERE / "Fig3_preview.png", dpi=200)
    plt.close(fig)

    (HERE / "Fig3_caption.txt").write_text(caption_text(REPLICATES) + "\n", encoding="utf-8")
    print(f"Wrote {pdf_path} (LaTeX-linked) and {HERE / 'Fig3_preview.png'}.")


if __name__ == "__main__":
    main()
