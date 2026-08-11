#!/usr/bin/env python3
"""Data generator for Supplementary Figure S2 (selected U0=8000 ABM means
versus the high-depth hierarchy across c, at R=4, canonical decomposition
Gamma=1, gamma=0, gamma_c=1, p_f=c).

Reads the pool-0 replicate trajectories for these 5 protocols from
../full_design_data/pool0_selected_trajectories.npz (written by
../run_full_design.py -- run that first: `python ../run_full_design.py
--threads 8`), summarizes U/U0, I/U0, M1/U0, m1=M1/I, and c*m1*I/U0 with
normal-approximation 95% CIs, and compares the ensemble mean against the
high-depth deterministic reference (K=40, checked against K=80). No new
ABM simulation is run here.

Run:
    python generate_data_S2.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FULL_DESIGN = HERE.parent / "full_design_data"
sys.path.insert(0, str(HERE.parent))
import trajectory_core_numba as tcn  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "abm_validation_u8000_core", HERE.parent / "generate_abm_mean_trajectory_error_convergence.py"
)
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)

R_VALUE = 4.0
C_GRID = (0., .25, .5, .75, 1.)
U0, I0 = 8000, 160
TAU = np.linspace(0.0, 5.0, 151)
SWITCH_TAU = 0.5
HIGH_K, CHECK_K = 40, 80


def protocol_id_for(cv: float) -> str:
    decomposition_id = "C" if cv == 1.0 else "F"
    return f"R{R_VALUE:g}_c{cv:g}_G1_{decomposition_id}"


def summarize(raw, cval):
    U = raw[:, :, 0] / U0
    I = raw[:, :, 1] / U0
    M1 = raw[:, :, 2] / U0
    with np.errstate(divide='ignore', invalid='ignore'):
        m1 = np.divide(raw[:, :, 2], raw[:, :, 1], out=np.full_like(raw[:, :, 2], np.nan), where=raw[:, :, 1] > 0)
    flux = cval * M1
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
    with np.load(FULL_DESIGN / "pool0_selected_trajectories.npz") as archive:
        selected = {name: archive[name] for name in archive.files}

    protocol_table = pd.read_csv(FULL_DESIGN / "protocol_table_U8000.csv").set_index("protocol_id")

    errors = []
    payload = {'tau': TAU}
    checks = []
    for cv in C_GRID:
        pid = protocol_id_for(cv)
        raw = selected[pid]  # (120, 151, 3): columns U, I, M1
        row = protocol_table.loc[pid]
        s = summarize(raw, cv)
        ref = _core.solve_high_depth(R_VALUE, cv, TAU, SWITCH_TAU, HIGH_K, I0 / U0)
        chk = _core.solve_high_depth(R_VALUE, cv, TAU, SWITCH_TAU, CHECK_K, I0 / U0)
        checks.append(float(np.max(np.abs(ref[:, :3] - chk[:, :3]))))
        mean = np.column_stack([s['U_mean'], s['I_mean'], s['M1_mean']])
        comp = _core.integrated_rmse(mean[None, :, :], ref[:, :3], TAU)[0]
        errors.append({
            'protocol_id': pid, 'R': R_VALUE, 'c': cv, 'Gamma': 1.0,
            'decomposition_id': row['decomposition_id'], 'beta': row['beta'],
            'gamma': row['gamma'], 'gamma_c': row['gamma_c'], 'p_f': row['p_f'],
            'n_replicates': raw.shape[0],
            'E_U': float(comp[0]), 'E_I': float(comp[1]), 'E_M1': float(comp[2]),
            'E_trajectory': float(np.linalg.norm(comp)),
        })
        tag = f'c{cv:g}'.replace('.', 'p')
        payload['raw_' + tag] = raw
        payload['reference_' + tag] = ref[:, :3]
        for k, v in s.items():
            payload[k + '_' + tag] = v
        print(f"done c={cv:g} (protocol {pid})", flush=True)

    np.savez_compressed(HERE / 'supp_S2_selected_production_data_U8000.npz', **payload)
    pd.DataFrame(errors).to_csv(HERE / 'supp_S2_selected_production_errors_U8000.csv', index=False)
    (HERE / 'supp_S2_high_depth_checks_U8000.json').write_text(
        json.dumps({'c_grid': list(C_GRID), 'max_abs_difference_K40_vs_K80': checks}, indent=2) + '\n'
    )
    print(pd.DataFrame(errors)[['c', 'E_trajectory']])


if __name__ == "__main__":
    main()
