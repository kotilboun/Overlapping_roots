#!/usr/bin/env python3
"""Figure generator for Supplementary Figure S4 (rate-scale collapse check).

Reads the data written by generate_data_S4.py (which must be run first);
does not run any new simulations.

Run:
    python generate_figure_S4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import u8000_common as c  # noqa: E402

ABM = '#2F6F9F'
DET = '#B65F32'
GAMMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
GAMMA_COLORS = ['#4C4CFF', '#2F6F9F', '#2E8B57', '#C08A1E', '#B65F32']

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": .8, "lines.linewidth": 1.1,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    with np.load(HERE / 'supp_S4_rate_scale_data_U8000.npz') as archive:
        payload = {name: archive[name] for name in archive.files}
    check = json.loads((HERE / 'supp_S4_high_depth_check_U8000.json').read_text())

    ref = payload['reference']
    I_ref = ref[:, 1]

    fig, axs = plt.subplots(1, 2, figsize=(6.85, 3.15))

    ax = axs[0]
    for Gamma, color in zip(GAMMA_GRID, GAMMA_COLORS):
        tag = f'G{Gamma:g}'.replace('.', 'p')
        t = payload['physical_time_' + tag]
        I = payload['mean_' + tag][:, 1]
        ax.plot(t, I, color=color, lw=1.1, label=rf'$\Gamma={Gamma:g}$')
    ax.set_xlabel(r'physical time $t$')
    ax.set_ylabel(r'$I(t)/S_0$ (ABM ensemble mean)')
    ax.set_title('(a) physical-time trajectories')
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=7.5, ncol=1, loc='upper right')

    ax = axs[1]
    ax.plot(c.TAU, I_ref, color=DET, lw=1.5, label=rf'high-depth hierarchy ($K={c.HIGH_K}$)', zorder=5)
    for Gamma, color in zip(GAMMA_GRID, GAMMA_COLORS):
        tag = f'G{Gamma:g}'.replace('.', 'p')
        I = payload['mean_' + tag][:, 1]
        ax.plot(c.TAU, I, color=color, lw=1.0, ls='--', alpha=.85, label=rf'$\Gamma={Gamma:g}$')
    ax.axvline(c.SWITCH_TAU, color='.55', lw=.75, ls=(0, (1.2, 2)), zorder=0)
    ax.set_xlabel(r'$\tau=\Gamma t$')
    ax.set_ylabel(r'$I(\tau)/S_0$')
    ax.set_title('(b) nondimensional collapse')
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=7, ncol=1, loc='upper right')

    fig.subplots_adjust(left=.11, right=.99, bottom=.16, top=.9, wspace=.32)
    for suf in ['pdf', 'png', 'tiff']:
        p = HERE / f'supp_S4_rate_scale_collapse_U8000.{suf}'
        if suf == 'pdf':
            fig.savefig(p)
        elif suf == 'png':
            fig.savefig(p, dpi=600)
        else:
            fig.savefig(p, dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    with Image.open(HERE / 'supp_S4_rate_scale_collapse_U8000.png') as im:
        im.convert('RGB').save(HERE / 'supp_S4_rate_scale_collapse_U8000.png', dpi=(600, 600))
    with Image.open(HERE / 'supp_S4_rate_scale_collapse_U8000.tiff') as im:
        im.convert('RGB').save(HERE / 'supp_S4_rate_scale_collapse_U8000.tiff', dpi=(600, 600), compression='tiff_lzw')

    caption = (
        'Supplementary Figure S4. Direct simulation of dimensional rate rescaling. Independently '
        'generated 120-realization ABM ensembles are shown for Gamma in {1/4,1/2,1,2,4} at R=4 and '
        'c=0.5, using the frequent-detection, partial-tracing decomposition (gamma=0, gamma_c=Gamma, '
        'p_f=c). Panel (a) shows physical-time trajectories of I(t)/S0 over 0<=t<=5/Gamma; proportional '
        'changes in the dimensional event rates alter the physical time scale. Panel (b) shows the same '
        'ABM ensemble means in tau=Gamma*t, where they collapse closely onto the common high-depth '
        'deterministic reference computed with '
        f'K={c.HIGH_K} (checked against K={c.CHECK_K}; maximum difference '
        f'{check["max_abs_difference_K40_vs_K80"]:.2e}). This is an implementation and nondimensionalization '
        'check rather than an additional biological sensitivity analysis. Protocols are pool 0 of the '
        'same 195-protocol, master-seed-20260804 design used throughout this Supplementary Material.'
    )
    (HERE / 'supp_S4_caption_U8000.txt').write_text(caption + '\n', encoding='utf-8')
    print("Figure S4 written.")


if __name__ == "__main__":
    main()
