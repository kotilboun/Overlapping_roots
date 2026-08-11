#!/usr/bin/env python3
"""Data generator for Supplementary Figure S3 (matched-c raw-rate
decomposition comparison at R=4, c=0.5, U0=8000).

Reads the pool-0 replicate trajectories for the two matched protocols
("A" = frequent detection/partial tracing = the F decomposition; "B" =
less-frequent detection/complete tracing = the L decomposition, both at
Gamma=1) from ../full_design_data/pool0_selected_trajectories.npz (written
by ../run_full_design.py -- run that first: `python ../run_full_design.py
--threads 8`), and compares against the common high-depth deterministic
reference. No new ABM simulation is run here; `u8000_common.matched_protocols()`
is used only for its protocol metadata/labels (Gamma, gamma, gamma_c, p_f),
which are algebraically identical to the F/L rows of the full 195-protocol
design at R=4, c=0.5, Gamma=1.

Run:
    python generate_data_S3.py
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FULL_DESIGN = HERE.parent / "full_design_data"
sys.path.insert(0, str(HERE.parent))
import u8000_common as c  # noqa: E402  (labels/metadata only, no simulation)

R_VALUE, C_VALUE = 4.0, 0.5
# u8000_common.matched_protocols() protocol_id -> full-design protocol_id
FULL_DESIGN_ID = {"A": "R4_c0.5_G1_F", "B": "R4_c0.5_G1_L"}


def summarize(raw):
    U = raw[:, :, 0] / c.U0
    I = raw[:, :, 1] / c.U0
    M1 = raw[:, :, 2] / c.U0
    with np.errstate(divide='ignore', invalid='ignore'):
        m1 = np.divide(raw[:, :, 2], raw[:, :, 1], out=np.full_like(raw[:, :, 2], np.nan), where=raw[:, :, 1] > 0)
    flux = .5 * M1
    outd = {}
    for name, v in [('U', U), ('I', I), ('M1', M1), ('m1', m1), ('flux', flux)]:
        if name == 'm1':
            mean = np.nanmean(v, axis=0)
            sd = np.nanstd(v, axis=0, ddof=1)
            n = np.sum(np.isfinite(v), axis=0)
            ci = 1.96 * sd / np.sqrt(np.maximum(n, 1))
        else:
            mean = v.mean(axis=0)
            sd = v.std(axis=0, ddof=1)
            ci = 1.96 * sd / np.sqrt(v.shape[0])
        outd[name + '_mean'] = mean
        outd[name + '_ci95'] = ci
    return outd


def main() -> None:
    ps = c.matched_protocols()
    with np.load(FULL_DESIGN / "pool0_selected_trajectories.npz") as archive:
        selected = {name: archive[name] for name in archive.files}

    ref = c.mod.solve_high_depth(R_VALUE, C_VALUE, c.TAU, c.SWITCH_TAU, c.HIGH_K, c.I0 / c.U0)
    rows = []
    payload = {'tau': c.TAU, 'reference': ref[:, :3]}
    for p in ps:
        raw = selected[FULL_DESIGN_ID[p.protocol_id]]
        s = summarize(raw)
        mean = np.column_stack([s['U_mean'], s['I_mean'], s['M1_mean']])
        comp = c.mod.integrated_rmse(mean[None, :, :], ref[:, :3], c.TAU)[0]
        rows.append({
            **asdict(p), 'n_replicates': raw.shape[0],
            'E_U': float(comp[0]), 'E_I': float(comp[1]), 'E_M1': float(comp[2]),
            'E_trajectory': float(np.linalg.norm(comp)),
        })
        payload['raw_' + p.protocol_id] = raw
        for k, v in s.items():
            payload[k + '_' + p.protocol_id] = v
        print(f"done protocol {p.protocol_id} ({FULL_DESIGN_ID[p.protocol_id]})", flush=True)

    np.savez_compressed(HERE / 'supp_S3_matched_c_data_U8000.npz', **payload)
    pd.DataFrame(rows).to_csv(HERE / 'supp_S3_matched_c_protocol_errors_U8000.csv', index=False)
    print(pd.DataFrame(rows)[['protocol_id', 'gamma', 'gamma_c', 'p_f', 'E_trajectory']])


if __name__ == "__main__":
    main()
