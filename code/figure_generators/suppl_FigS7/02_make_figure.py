#!/usr/bin/env python3
"""Step 2/2 for Figure 6: draw the figure from the cached ensemble closure terms.

Reads only ``data/*`` (written by ``01_generate_data.py``) and draws the
two-row, four-panel local-closure-validation figure (top: closure-supplied vs.
ABM-measured local term; bottom: signed-defect histograms). The PDF is written
directly to the manuscript's ``figures/Fig6.pdf`` -- the exact path
``\\includegraphics{figures/Fig6.pdf}`` in manuscript.tex uses -- so re-running
this script is the only step needed to refresh what LaTeX renders. It never
re-simulates the ABM.

Run (after 01_generate_data.py has populated data/):
    python 02_make_figure.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
# Canonical location LaTeX includes; this script is the only writer of Fig6.pdf.
MANUSCRIPT_FIGURES_DIR = HERE.parents[1] / "figures"

R_VALUES = (1.0, 1.5, 2.0, 4.0)
C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
R_MARKERS = {1.0: "o", 1.5: "s", 2.0: "^", 4.0: "D"}
CLOSURES = ("algebraic_qss0", "dynamic_K1", "dynamic_K2", "dynamic_K3")
TITLES = (
    "zeroth-order\nalgebraic QSS",
    "dynamic tail\n$K=1$",
    "dynamic tail\n$K=2$",
    "dynamic tail\n$K=3$",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_terms() -> dict[str, dict[str, np.ndarray]]:
    with np.load(DATA_DIR / "snapshot_terms.npz") as archive:
        flat = {name: archive[name] for name in archive.files}
    terms: dict[str, dict[str, np.ndarray]] = {closure: {} for closure in CLOSURES}
    for key, array in flat.items():
        for closure in CLOSURES:
            prefix = f"{closure}_"
            if key.startswith(prefix):
                terms[closure][key[len(prefix):]] = array
    return terms


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure_atomically(fig: plt.Figure, path: Path, **kwargs: Any) -> None:
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        fig.savefig(tmp, format=path.suffix.lstrip("."), **kwargs)
        if path.suffix.lower() == ".pdf":
            data = tmp.read_bytes()
            if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-64:]:
                raise OSError(f"PDF verification failed: {tmp}")
        else:
            with Image.open(tmp) as image:
                image.verify()
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def format_histogram_tick(value: float, _position: float) -> str:
    if abs(value) < 1.0e-15:
        return "0"
    if abs(value) < 1.0e-3:
        return f"{value:.1e}"
    decimals = max(0, int(1 - np.floor(np.log10(abs(value)))))
    return f"{value:.{decimals}f}"


def make_figure(
    terms: dict[str, dict[str, np.ndarray]],
    histogram_rows: list[dict[str, Any]],
) -> plt.Figure:
    publication_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 4.65), gridspec_kw={"height_ratios": [2.15, 1.0]})
    fig.subplots_adjust(left=0.09, right=0.92, top=0.82, bottom=0.12, wspace=0.36, hspace=0.40)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)

    for panel, (closure, title) in enumerate(zip(CLOSURES, TITLES)):
        ax = axes[0, panel]
        data = terms[closure]
        for R in R_VALUES:
            for c in C_VALUES:
                sel = (data["R"] == R) & (data["c"] == c)
                closure_term = data["x"][sel]
                abm_term = data["y"][sel]
                tau = data["tau"][sel]
                if len(closure_term) == 0:
                    continue
                order = np.argsort(tau)
                x = abm_term[order]
                y = closure_term[order]
                color = cmap(norm(c))
                ax.plot(x, y, color=color, linewidth=0.65, alpha=0.68, zorder=1)
                step = max(1, len(x) // 12)
                ax.plot(
                    x[::step], y[::step], linestyle="none", marker=R_MARKERS[R],
                    markersize=2.8, markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=0.25, alpha=0.85, zorder=2,
                )

        combined = np.concatenate([data["x"][np.isfinite(data["x"])], data["y"][np.isfinite(data["y"])]])
        low = min(0.0, float(combined.min()))
        high = max(0.0, float(combined.max()))
        span = max(high - low, 1e-3)
        limits = (low - 0.06 * span, high + 0.06 * span)
        ax.plot(limits, limits, color="0.30", linewidth=0.9, zorder=0)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.grid(color="0.84", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, pad=3.0)
        ax.text(
            -0.18, 1.20, f"({chr(97 + panel)})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11, fontweight="bold", clip_on=False,
        )

        hist_ax = axes[1, panel]
        delta = data["x"] - data["y"]
        selected = np.isfinite(delta) & (data["c"] > 0.0)
        values = delta[selected]
        hist_ax.hist(values, bins=18, color="0.72", edgecolor="0.35", linewidth=0.5)
        hist_ax.axvline(0.0, color="0.20", linewidth=0.9, linestyle="--")
        tick_row = histogram_rows[panel]
        ticks = [
            float(tick_row["tick_mean_minus_2sd"]),
            float(tick_row["tick_mean"]),
            float(tick_row["tick_mean_plus_2sd"]),
        ]
        hist_ax.xaxis.set_major_locator(FixedLocator(ticks))
        hist_ax.xaxis.set_major_formatter(FuncFormatter(format_histogram_tick))
        hist_ax.tick_params(axis="x", labelsize=7.8, pad=1.5)
        for label in hist_ax.get_xticklabels():
            label.set_rotation(35)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
        hist_ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        hist_ax.grid(color="0.88", linewidth=0.5, alpha=0.7)
        hist_ax.set_axisbelow(True)
        histogram_labels = (r"$\Delta_Z$", r"$\Delta_{K1}$", r"$\Delta_{K2}$", r"$\Delta_{K3}$")
        if panel == 0:
            hist_ax.set_ylabel("count")
        hist_ax.set_xlabel(histogram_labels[panel])

    legend_handles = [
        Line2D(
            [], [], marker=R_MARKERS[R], linestyle="none", markersize=5,
            markerfacecolor="0.25", markeredgecolor="0.25", label=rf"$R={R:g}$",
        )
        for R in R_VALUES
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.50, 0.93), ncol=4,
        frameon=False, handletextpad=0.30, columnspacing=0.95, handlelength=0.8, borderaxespad=0.0,
    )

    cax = fig.add_axes([0.935, 0.50, 0.016, 0.26])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.set_title(r"$c$", pad=4.0)
    cb.set_ticks(np.linspace(0.0, 1.0, 6))

    axes[0, 0].set_ylabel("closure-supplied local term")
    fig.text(0.47, 0.398, "ABM-measured local term", ha="center", va="center", fontsize=10)

    return fig


def caption_text(protocol: dict[str, Any]) -> str:
    return (
        "Fig. 6 Local closure terms evaluated along ensemble-mean "
        "finite-population ABM trajectories. Each colored path comprises "
        f"post-activation snapshots (tau >= {protocol['tau_on']:g}) obtained after "
        f"averaging {protocol['replicates']} independent realizations within each "
        "parameter cell; color denotes c and marker shape denotes R. In the top "
        "row, the horizontal coordinate is the ABM-measured local term and the "
        "vertical coordinate is the closure-supplied local term; the diagonal "
        "denotes equality. (a) Zeroth-order algebraic QSS closure: "
        r"c\overline{m}_1^{\mathrm{ABM}} is compared with "
        r"c\widehat{m}_1^{(0)}(\overline{R}_S^{\mathrm{ABM}}). "
        "(b-d) Dynamic tail closures K=1,2,3: "
        r"c[\overline{m}_1^{\mathrm{ABM}}\overline{m}_K^{\mathrm{ABM}}-"
        r"\overline{m}_{K+1}^{\mathrm{ABM}}] is compared with "
        r"c[\overline{m}_1^{\mathrm{ABM}}\overline{m}_K^{\mathrm{ABM}}-"
        r"\Phi_{K+1}^{\mathrm{tail}}(\overline{R}_S^{\mathrm{ABM}};"
        r"\overline{m}_K^{\mathrm{ABM}})]. "
        "All quantities in a snapshot are evaluated at the same ensemble-mean "
        r"state, with \overline{R}_S^{\mathrm{ABM}}="
        r"R\overline{S}^{\mathrm{ABM}}/S_0. "
        "The bottom row shows descriptive snapshot distributions of the signed "
        "closure-supplied-minus-ABM-measured differences: "
        r"\Delta_Z=c[\widehat{m}_1^{(0)}-"
        r"\overline{m}_1^{\mathrm{ABM}}] and "
        r"\Delta_{K}=c[\overline{m}_{K+1}^{\mathrm{ABM}}-"
        r"\Phi_{K+1}^{\mathrm{tail}}] for K=1,2,3, labeled "
        r"\Delta_Z,\Delta_{K1},\Delta_{K2}, and \Delta_{K3}. "
        "Histograms pool the displayed ensemble-mean snapshots with c>0; their "
        "three x-axis ticks are the mean minus two population standard "
        "deviations, the mean, and the mean plus two population standard "
        "deviations, rounded to two significant digits. Simulations use "
        f"S_0={protocol['U0']}, I_0={protocol['I0']}, "
        r"R\in\{1,1.5,2,4\}, c\in\{0,0.25,0.5,0.75,1\}, "
        f"and {protocol['num_times']} observation times on "
        f"0 <= tau <= {protocol['tau_end']:g}, with "
        r"(gamma/Gamma,gamma_c/Gamma,p_f)=(0,1,c). "
        "Normalized genealogical coordinates are averaged conditionally over "
        "realizations with positive infectious count; time-resolved alive risk "
        "sets are supplied with the figure data.\n"
    )


def main() -> None:
    if not (DATA_DIR / "snapshot_terms.npz").exists():
        raise FileNotFoundError(f"{DATA_DIR} is missing cached data; run 01_generate_data.py first.")

    terms = load_terms()
    histogram_rows = read_csv(DATA_DIR / "histogram_tick_summary.csv")
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))

    fig = make_figure(terms, histogram_rows)

    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = MANUSCRIPT_FIGURES_DIR / "Fig6.pdf"
    save_figure_atomically(fig, pdf_path, facecolor="white")
    fig.savefig(HERE / "Fig6_preview.png", dpi=200, facecolor="white")
    plt.close(fig)

    (HERE / "Fig6_caption.txt").write_text(caption_text(manifest["protocol"]), encoding="utf-8")
    print(f"Wrote {pdf_path} (LaTeX-linked) and {HERE / 'Fig6_preview.png'}.")


if __name__ == "__main__":
    main()
