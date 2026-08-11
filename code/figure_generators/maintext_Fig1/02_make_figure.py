#!/usr/bin/env python3
"""Step 2/2 for Figure 1: draw the figure from the cached, checked forest data.

Reads ``forest_spec.json`` and ``data/forest_check.json`` (written by
``01_generate_data.py``) and draws the three-panel bookkeeping figure. The PDF is
written directly to the manuscript's ``figures/Fig1.pdf`` -- the exact path
``\\includegraphics{figures/Fig1.pdf}`` in manuscript.tex uses -- so re-running this
script is the only step needed to refresh what LaTeX renders.

Run (after 01_generate_data.py has produced data/forest_check.json):
    python 02_make_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "forest_spec.json"
CHECK_PATH = HERE / "data" / "forest_check.json"
# Canonical location LaTeX includes; this script is the only writer of Fig1.pdf.
MANUSCRIPT_FIGURES_DIR = HERE.parents[1] / "figures"

CAPTION = (
    "Fig. 1 Root-centered bookkeeping in one physical active transmission forest. "
    "(a) A single physical active genealogy, with active individual i, its active child j, "
    "and the physical subtree descended from i indicated by the dashed boundary. "
    "(b) The same active individuals viewed in coordinates rooted at i, giving "
    "D_i^(1)=3, D_i^(2)=2, and D_i^(3)=2. "
    "The dotted boundary encloses the embedded subtree rooted at j. "
    "(c) Re-indexing the same physical individuals relative to root j gives "
    "D_j^(1)=2 and D_j^(2)=2. "
    "The local views overlap but do not duplicate individuals: "
    "a person is removed physically at most once, although that event can remove several "
    "root-descendant relationships from the finite-forest counts "
    "M_{k,N}=sum_{i in A_N} D_i^(k)."
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def setup_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        -0.02,
        1.01,
        rf"$\bf{{{label}}}$ {title}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        clip_on=False,
    )


def draw_node(
    ax: plt.Axes,
    xy: tuple[float, float],
    label: str,
    *,
    edge: str = "0.25",
    face: str = "white",
    lw: float = 0.9,
) -> None:
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=260.0,
        facecolors=face,
        edgecolors=edge,
        linewidths=lw,
        zorder=3,
    )
    ax.text(xy[0], xy[1], rf"${label}$", ha="center", va="center", fontsize=8.5, zorder=4)


def draw_directed_edge(
    ax: plt.Axes,
    parent: tuple[float, float],
    child: tuple[float, float],
    *,
    color: str,
) -> None:
    ax.annotate(
        "",
        xy=child,
        xytext=parent,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 0.8,
            "mutation_scale": 7,
            "shrinkA": 9,
            "shrinkB": 9,
        },
        zorder=1,
    )


def count_block(
    ax: plt.Axes,
    x: float,
    y: float,
    root: str,
    counts: dict[str, int],
) -> None:
    for row, (depth, value) in enumerate(counts.items()):
        yy = y - row * 0.095
        ax.text(
            x,
            yy,
            rf"$D_{{{root}}}^{{({depth})}}={value}$",
            ha="left",
            va="center",
            fontsize=8.8,
        )


def make_figure(spec: dict[str, Any], check: dict[str, Any], out_dir: Path) -> Path:
    node_by_id = {str(node["id"]): node for node in spec["nodes"]}
    positions = {
        node_id: (float(node["x"]), float(node["y"]))
        for node_id, node in node_by_id.items()
    }
    edges = [(str(parent), str(child)) for parent, child in check["directed_active_edges"]]
    highlight = spec["display"]["highlight"]

    fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.35), constrained_layout=True)
    for ax in axes:
        setup_panel(ax)

    panel_specs = [
        (
            axes[0],
            "(a)",
            "one physical active forest",
            list(node_by_id),
            edges,
        ),
        (
            axes[1],
            "(b)",
            r"bookkeeping rooted at $i$",
            check["selected_root_checks"]["i"]["local_nodes"],
            None,
        ),
        (
            axes[2],
            "(c)",
            r"bookkeeping rooted at $j$",
            check["selected_root_checks"]["j"]["local_nodes"],
            None,
        ),
    ]

    for ax, letter, title, visible_nodes, panel_edges in panel_specs:
        panel_title(ax, letter, title)
        visible = set(visible_nodes)
        active_edges = panel_edges or [
            edge for edge in edges if edge[0] in visible and edge[1] in visible
        ]
        for parent, child in active_edges:
            draw_directed_edge(
                ax,
                positions[parent],
                positions[child],
                color="0.35" if ax is axes[0] else "0.15",
            )
        for node_id in visible_nodes:
            styling = highlight.get(node_id, {})
            draw_node(
                ax,
                positions[node_id],
                node_id,
                edge=styling.get("edge", "0.25"),
                face=styling.get("face", "white"),
                lw=1.2 if node_id in highlight else 0.9,
            )

    # Panel (a) marks the physical subtree rooted at i with a dashed boundary.
    axes[0].add_patch(
        patches.Rectangle(
            (0.18, -0.005),
            0.46,
            0.75,
            fill=False,
            edgecolor="black",
            linewidth=0.8,
            linestyle=(0, (5, 4)),
            zorder=0,
        )
    )
    axes[0].text(0.68, 0.66, r"selected root $i$", ha="left", va="center", fontsize=7)
    axes[0].text(0.48, 0.36, r"active child $j$", ha="left", va="center", fontsize=7)

    # In the i-rooted bookkeeping view, mark the overlapping j-rooted subtree.
    # A dotted style distinguishes this coordinate-view boundary from the
    # dashed physical-subtree boundary in panel (a).
    axes[1].add_patch(
        patches.Rectangle(
            (0.255, -0.005),
            0.29,
            0.535,
            fill=False,
            edgecolor="black",
            linewidth=0.8,
            linestyle=(0, (1.2, 2.4)),
            zorder=0,
        )
    )

    axes[1].text(0.18, 0.46, "children", ha="right", va="center", fontsize=6.8)
    axes[1].text(0.18, 0.26, "grandchildren", ha="right", va="center", fontsize=6.8)
    axes[1].text(0.18, 0.06, "great-\ngrandchildren", ha="right", va="center", fontsize=6.8)
    count_block(
        axes[1],
        0.68,
        0.80,
        "i",
        check["selected_root_checks"]["i"]["observed"],
    )

    axes[2].text(0.23, 0.26, "children", ha="right", va="center", fontsize=7)
    axes[2].text(0.23, 0.06, "grandchildren", ha="right", va="center", fontsize=7)
    count_block(
        axes[2],
        0.52,
        0.70,
        "j",
        check["selected_root_checks"]["j"]["observed"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "Fig1.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(HERE / "Fig1_preview.png", bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf_path


def main() -> None:
    if not CHECK_PATH.exists():
        raise FileNotFoundError(
            f"{CHECK_PATH} is missing; run 01_generate_data.py first."
        )
    set_style()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    check = json.loads(CHECK_PATH.read_text(encoding="utf-8"))
    if not check.get("all_checks_passed"):
        raise RuntimeError("Cached forest_check.json did not pass its checks.")
    pdf_path = make_figure(spec, check, MANUSCRIPT_FIGURES_DIR)
    (HERE / "Fig1_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    print(f"Wrote {pdf_path} (LaTeX-linked) and {HERE / 'Fig1_preview.png'}.")


if __name__ == "__main__":
    main()
