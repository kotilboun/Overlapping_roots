#!/usr/bin/env python3
"""Figure generator for Supplementary Figure S3.

Reads the data files written by generate_data_S3.py (which must be run
first) and produces the publication PDF/PNG/TIFF and frozen caption. Does
not run any new simulations.

Run:
    python generate_figure_S3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import u8000_common as c  # noqa: E402

ABM = '#2F6F9F'
DET = '#B65F32'

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": .8, "lines.linewidth": 1,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    ps = c.matched_protocols()
    rows = pd.read_csv(HERE / 'supp_S3_matched_c_protocol_errors_U8000.csv').to_dict('records')
    with np.load(HERE / 'supp_S3_matched_c_data_U8000.npz') as archive:
        payload = {name: archive[name] for name in archive.files}

    ref = payload['reference']
    M1 = ref[:, 2]
    I = ref[:, 1]
    det = {'S': ref[:, 0], 'I': I, 'm1': np.divide(M1, I, out=np.zeros_like(M1), where=I > 0), 'flux': .5 * M1}

    sums = []
    for p in ps:
        s = {k[:-(len(p.protocol_id) + 1)]: v for k, v in payload.items()
             if k.endswith('_' + p.protocol_id) and not k.startswith('raw_')}
        sums.append(s)

    row_specs = (
        ('I', r'$I(\tau)/S_0$', (0, .46), (0, .2, .4)),
        ('S', r'$S(\tau)/S_0$', (0, 1.04), (0, .5, 1)),
        ('m1', r'$m_1(\tau)$', (-.02, .82), (0, .4, .8)),
        ('flux', r'$c\,m_1(\tau)I(\tau)/S_0$', (-.004, .185), (0, .1)),
    )
    idx = np.unique(np.concatenate([np.arange(0, len(c.TAU), 6), [int(np.argmin(np.abs(c.TAU - c.SWITCH_TAU)))], [len(c.TAU) - 1]]))
    fig, axes = plt.subplots(4, 2, figsize=(6.85, 5.2), sharex=True, sharey='row', squeeze=False)
    for col, (p, s) in enumerate(zip(ps, sums)):
        for row, (name, yl, ylim, yt) in enumerate(row_specs):
            ax = axes[row, col]
            sk = 'U' if name == 'S' else name
            mean = s[sk + '_mean']
            ci = s[sk + '_ci95']
            valid = np.isfinite(mean[idx]) & np.isfinite(ci[idx])
            ii = idx[valid]
            ax.errorbar(c.TAU[ii], mean[ii], yerr=ci[ii], fmt='o', markersize=2.1, markeredgewidth=0,
                        color=ABM, ecolor=ABM, elinewidth=.65, capsize=1.25, capthick=.65,
                        label='ABM mean ± 95% CI' if row == 0 and col == 0 else None, zorder=4)
            ax.plot(c.TAU, det[name], color=DET, lw=1.25,
                     label=rf'high-depth hierarchy ($K={c.HIGH_K}$)' if row == 0 and col == 0 else None, zorder=3)
            ax.axvline(c.SWITCH_TAU, color='.55', lw=.75, ls=(0, (1.2, 2)), zorder=0)
            ax.set_xlim(-.03, 5.03); ax.set_ylim(*ylim); ax.set_yticks(yt); ax.set_xticks([0, 2, 4])
            ax.grid(color='.82', lw=.45, alpha=.45); ax.tick_params(direction='out', pad=2)
            if col == 0:
                ax.set_ylabel(yl, labelpad=4)
            if row == 3:
                ax.set_xlabel(r'$\tau$', labelpad=2)
            if row == 0:
                ax.set_title(p.label + '\n' + rf'$\Gamma={p.Gamma:g}$, $\gamma={p.gamma:g}$, $\gamma_c={p.gamma_c:g}$, $p_f={p.p_f:g}$', pad=3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='upper center', ncol=2, handlelength=2.8, columnspacing=2, handletextpad=.7, bbox_to_anchor=(.5, .995))
    fig.subplots_adjust(left=.125, right=.995, bottom=.08, top=.835, wspace=.3, hspace=.2)
    for suf in ['pdf', 'png', 'tiff']:
        pth = HERE / f'supp_S3_matched_c_raw_rate_comparison_U8000.{suf}'
        if suf == 'pdf':
            fig.savefig(pth)
        elif suf == 'png':
            fig.savefig(pth, dpi=600)
        else:
            fig.savefig(pth, dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    with Image.open(HERE / 'supp_S3_matched_c_raw_rate_comparison_U8000.png') as im:
        im.convert('RGB').save(HERE / 'supp_S3_matched_c_raw_rate_comparison_U8000.png', dpi=(600, 600))
    with Image.open(HERE / 'supp_S3_matched_c_raw_rate_comparison_U8000.tiff') as im:
        im.convert('RGB').save(HERE / 'supp_S3_matched_c_raw_rate_comparison_U8000.tiff', dpi=(600, 600), compression='tiff_lzw')

    caption = (
        'Supplementary Figure S3. Effect of changing the removal-and-tracing decomposition while keeping '
        'the nondimensional parameters fixed, shown for R=4, c=0.5, S0=8000, I0=160, and 120 ABM '
        'realizations per protocol. Both protocols use Γ=1 and β=RΓ/S0=0.0005, and therefore share the '
        'same nondimensional deterministic reference. The left column uses frequent detection with partial '
        'tracing (γ=0, γ_c=1, p_f=0.5), whereas the right column uses less-frequent detection with complete '
        'tracing (γ=0.5, γ_c=0.5, p_f=1). Rows show I(τ)/S0, S(τ)/S0, m1(τ), and c m1(τ)I(τ)/S0. Blue '
        'points show ABM replicate means with 95% confidence-interval error bars, and orange curves show '
        'the common high-depth deterministic reference computed with K=40. The vertical dotted line marks '
        'activation of forward tracing at τ=0.5. The confidence intervals are narrow and may be partly '
        'obscured by the ABM markers.'
    )
    (HERE / 'supp_S3_caption_U8000.txt').write_text(caption + '\n', encoding='utf-8')
    print(pd.DataFrame(rows)[['protocol_id', 'gamma', 'gamma_c', 'p_f', 'E_trajectory']])


if __name__ == "__main__":
    main()
