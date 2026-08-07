#!/usr/bin/env python3
"""Figure generator for Supplementary Figure S1 (= ESM_1.tex Fig. OR1.1).

Reads the tables written by generate_data_S1.py (which must be run first)
and produces the publication PDF/PNG/TIFF and frozen caption. Does not run
any new simulations.

Run:
    python generate_figure_S1.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": .8, "lines.linewidth": 1,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    pop = pd.read_csv(HERE / "population_size_convergence.csv")
    rep = pd.read_csv(HERE / "replicate_convergence_U8000.csv")

    fig, axs = plt.subplots(1, 2, figsize=(6.85, 3.15))

    ax = axs[0]
    x = pop.U0.to_numpy(dtype=float)
    y = pop["median"].to_numpy()
    err = np.vstack([y - pop.q10.to_numpy(), pop.q90.to_numpy() - y])
    ax.errorbar(x, y, yerr=err, fmt='o-', color='#2F6F9F', capsize=2, markersize=3)
    ax.plot(x, y[0] * (x / x[0]) ** -.5, '--', color='.35', lw=1, label=r'reference slope $S_0^{-1/2}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'population scale $S_0$')
    ax.set_ylabel(r'ensemble-mean trajectory error $E_{\mathrm{mean}}$')
    ax.set_title('(a) population-size convergence')
    ax.grid(alpha=.25); ax.legend(frameon=False, loc='lower left')

    ax = axs[1]
    x = rep.n_replicates.to_numpy(dtype=float)
    y = rep["median"].to_numpy()
    err = np.vstack([y - rep.q10.to_numpy(), rep.q90.to_numpy() - y])
    ax.errorbar(x, y, yerr=err, fmt='o-', color='#2F6F9F', capsize=2, markersize=3)
    ax.plot(x, y[0] * (x / x[0]) ** -.5, '--', color='.35', lw=1, label=r'reference slope $n^{-1/2}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('replicates in ABM mean')
    ax.set_title(r'(b) replicate convergence at $S_0=8000$')
    ax.grid(alpha=.25); ax.legend(frameon=False, loc='lower left')

    fig.subplots_adjust(left=.11, right=.99, bottom=.18, top=.9, wspace=.3)
    for suf in ['pdf', 'png', 'tiff']:
        p = HERE / f'supp_S1_convergence_U8000.{suf}'
        if suf == 'pdf':
            fig.savefig(p)
        elif suf == 'png':
            fig.savefig(p, dpi=600)
        else:
            fig.savefig(p, dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    with Image.open(HERE / 'supp_S1_convergence_U8000.png') as im:
        im.convert('RGB').save(HERE / 'supp_S1_convergence_U8000.png', dpi=(600, 600))
    with Image.open(HERE / 'supp_S1_convergence_U8000.tiff') as im:
        im.convert('RGB').save(HERE / 'supp_S1_convergence_U8000.tiff', dpi=(600, 600), compression='tiff_lzw')

    caption = (
        'Supplementary Figure S1. Convergence of the ABM ensemble-mean trajectory error to the '
        'high-depth deterministic reference. Panel (a) summarizes one 120-realization ensemble for '
        'each of 195 raw-rate protocols at each population scale; for S0=8000, this is the first '
        'prespecified pool (pool 0) among the 30 independently generated pools. Panel (b) uses 30 '
        'independent pools per protocol at S0=8000, yielding 5850 estimates for every n, including '
        'n=120. Points show medians and error bars the 10th and 90th percentiles. Dashed lines '
        'indicate reference slopes proportional to S0^{-1/2} and n^{-1/2}. The deterministic '
        'reference retained moments through K=40 and was checked against K=80.'
    )
    (HERE / 'supp_S1_caption_U8000.txt').write_text(caption + '\n', encoding='utf-8')
    print("Figure S1 written.")


if __name__ == "__main__":
    main()
