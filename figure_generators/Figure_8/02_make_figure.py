#!/usr/bin/env python3
"""Step 2/2 for Figure 8: draw representative R=4 trajectories vs. dynamic closures.

Reads only the cached ``data/abm_trajectory_summary.npz`` and
``data/closure_trajectories.npz`` (written by ``01_generate_data.py``) and
draws the four-row, five-column trajectory figure. The PDF is written
directly to the manuscript's ``figures/Fig8.pdf`` -- the exact path
``\\includegraphics{figures/Fig8.pdf}`` in sn-article.tex uses -- so
re-running this script is the only step needed to refresh what LaTeX
renders. It never re-simulates the ABM or re-solves the closures.

Run (after 01_generate_data.py has populated data/):
    python 02_make_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MANUSCRIPT_FIGURES_DIR = HERE.parents[1] / "figures"

C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
DYNAMIC_ORDERS = (1, 2, 3)
ROW_SPECS = (
    ("i", r"$i(\tau)=I(\tau)/S_0$", (0, 0.46), (0, 0.2, 0.4)),
    ("u", r"$s(\tau)=S(\tau)/S_0$", (0, 1.04), (0, 0.5, 1)),
    ("m1", r"$m_1(\tau)$", (-0.02, 0.82), (0, 0.4, 0.8)),
    ("flux", r"$c\,m_1(\tau)i(\tau)$", (-0.004, 0.185), (0, 0.1)),
)
ABM_COLOR = "#276B9A"
K_COLORS = {1: "#C0643B", 2: "#3F8754", 3: "#72599B"}
K_STYLES = {1: (0, (5, 2.5)), 2: "-.", 3: ":"}
SWITCH_TAU = 0.5


def archive_key(c: float) -> str:
    return f"c{c:g}".replace(".", "p")


def make_rgb(path: Path, compression: str | None = None) -> None:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        options: dict[str, object] = {"dpi": (600, 600)}
        if compression:
            options["compression"] = compression
        rgb.save(path, **options)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def main() -> None:
    summary_path = DATA_DIR / "abm_trajectory_summary.npz"
    closures_path = DATA_DIR / "closure_trajectories.npz"
    if not summary_path.exists() or not closures_path.exists():
        raise FileNotFoundError(f"{DATA_DIR} is missing cached data; run 01_generate_data.py first.")

    with np.load(summary_path) as summary, np.load(closures_path) as closures:
        times = summary["times"]
        set_style()
        fig, axes = plt.subplots(4, 5, figsize=(6.85, 5.4), sharex=True, sharey="row", squeeze=False)
        for column, c in enumerate(C_VALUES):
            key = archive_key(c)
            for row_index, (variable, ylabel, ylim, yticks) in enumerate(ROW_SPECS):
                axis = axes[row_index, column]
                mean = summary[f"abm_{variable}_mean_{key}"]
                q025 = summary[f"abm_{variable}_q025_{key}"]
                q975 = summary[f"abm_{variable}_q975_{key}"]
                axis.fill_between(
                    times, q025, q975, color=ABM_COLOR, alpha=0.15, linewidth=0,
                    label="ABM replicate 95% range" if column == 0 and row_index == 0 else None,
                )
                axis.plot(
                    times, mean, color=ABM_COLOR, linewidth=1.35,
                    label="ABM mean" if column == 0 and row_index == 0 else None,
                )
                for order in DYNAMIC_ORDERS:
                    axis.plot(
                        times, closures[f"dynamic_K{order}_{variable}_{key}"],
                        color=K_COLORS[order], linestyle=K_STYLES[order], linewidth=1.0,
                        label=rf"dynamic $K={order}$" if column == 0 and row_index == 0 else None,
                    )
                axis.axvline(SWITCH_TAU, color="0.55", linewidth=0.7, linestyle=(0, (1.2, 2.0)))
                axis.set_xlim(-0.03, 5.03)
                axis.set_ylim(*ylim)
                axis.set_yticks(yticks)
                axis.set_xticks([0, 2, 4])
                axis.grid(color="0.84", linewidth=0.45, alpha=0.5)
                axis.tick_params(direction="out", pad=2)
                if column == 0:
                    axis.set_ylabel(ylabel, labelpad=4)
                if row_index == 3:
                    axis.set_xlabel(r"$\tau$", labelpad=2)
                if row_index == 0:
                    axis.set_title(rf"$c={c:g}$", pad=3)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.993), ncol=3,
            frameon=False, handlelength=2.6, columnspacing=1.8, handletextpad=0.6,
        )
        fig.subplots_adjust(left=0.115, right=0.995, bottom=0.08, top=0.88, wspace=0.28, hspace=0.2)

        MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = MANUSCRIPT_FIGURES_DIR / "Fig8.pdf"
        fig.savefig(pdf_path, metadata={"Title": "Figure 8: representative R=4 trajectories vs. dynamic closures", "Creator": Path(__file__).name})
        png_path = pdf_path.with_suffix(".png")
        tiff_path = pdf_path.with_suffix(".tiff")
        fig.savefig(png_path, dpi=600)
        fig.savefig(tiff_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(HERE / "Fig8_preview.png", dpi=200)
        plt.close(fig)
        make_rgb(png_path)
        make_rgb(tiff_path, "tiff_lzw")

    caption = (
        "Fig. 8 Representative finite-pool trajectories for the depth-K dynamic closures at R=4. "
        "Columns show c=0, 0.25, 0.5, 0.75, and 1; rows show i=I/S_0, s=S/S_0, m_1=M_1/I, and the "
        "normalized deterministic tracing flux c m_1 i. Blue curves are the sample ensemble-mean ABM "
        "trajectory and pale bands are pointwise 2.5th-97.5th replicate percentiles across 120 "
        "independent realizations. Orange dashed, green dash-dotted, and purple dotted curves are the "
        "dynamic K=1, 2, and 3 closures (Sect. 7.1). The vertical dotted line marks activation of "
        "forward tracing at tau=0.5. Simulations use S_0=8000, I_0=160, and 151 observation times on "
        "0<=tau<=5, with (gamma/Gamma,gamma_c/Gamma,p_f)=(0,1,c). The trajectories shown are the same "
        "R=4 ABM ensemble underlying the E_C errors reported in Fig. 7.\n"
    )
    (HERE / "Fig8_caption.txt").write_text(caption, encoding="utf-8")
    print(f"Wrote {pdf_path} (LaTeX-linked) and {HERE / 'Fig8_preview.png'}.")


if __name__ == "__main__":
    main()
