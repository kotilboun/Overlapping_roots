#!/usr/bin/env python3
"""Data generator for Supplementary Figure S4 (rate-scale collapse check).

Reads the pool-0 replicate trajectories for the five rate-scale protocols
(R=4, c=0.5, Gamma in {0.25,0.5,1,2,4}, frequent-detection/partial-tracing
decomposition) from ../full_design_data/pool0_selected_trajectories.npz
(written by ../run_full_design.py -- run that first: `python
../run_full_design.py --threads 8`). No new ABM simulation is run here.

This supersedes the earlier standalone reconstruction of S4 (which used an
independent, undocumented seed because no original S4 source code survived
in this project): S4 is now part of the same 195-protocol/master-seed-20260804
design as S1-S3, so it is no longer a special case.

Run:
    python generate_data_S4.py
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
import u8000_common as c  # noqa: E402  (solve_high_depth / integrated_rmse only)

R = 4.0
C_VALUE = 0.5
GAMMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)


def full_design_id(Gamma: float) -> str:
    return f"R{R:g}_c{C_VALUE:g}_G{Gamma:g}_F"


def main() -> None:
    ref = c.mod.solve_high_depth(R, C_VALUE, c.TAU, c.SWITCH_TAU, c.HIGH_K, c.I0 / c.U0)
    chk = c.mod.solve_high_depth(R, C_VALUE, c.TAU, c.SWITCH_TAU, c.CHECK_K, c.I0 / c.U0)
    max_check_diff = float(np.max(np.abs(ref[:, :3] - chk[:, :3])))

    with np.load(FULL_DESIGN / "pool0_selected_trajectories.npz") as archive:
        selected = {name: archive[name] for name in archive.files}
    protocol_table = pd.read_csv(FULL_DESIGN / "protocol_table_U8000.csv").set_index("protocol_id")

    rows = []
    payload = {"tau": c.TAU, "reference": ref[:, :3]}
    for Gamma in GAMMA_GRID:
        pid = full_design_id(Gamma)
        raw = selected[pid]  # (120, 151, 3): columns U, I, M1
        info = protocol_table.loc[pid]
        norm = raw.astype(float) / c.U0
        mean = norm.mean(axis=0)
        comp = c.mod.integrated_rmse(mean[None, :, :], ref[:, :3], c.TAU)[0]
        rows.append({
            "protocol_id": pid, "R": R, "c": C_VALUE, "Gamma": Gamma,
            "beta": info["beta"], "gamma": info["gamma"], "gamma_c": info["gamma_c"], "p_f": info["p_f"],
            "n_replicates": raw.shape[0],
            "E_U": float(comp[0]), "E_I": float(comp[1]), "E_M1": float(comp[2]),
            "E_trajectory": float(np.linalg.norm(comp)),
        })
        tag = f"G{Gamma:g}".replace(".", "p")
        payload["mean_" + tag] = mean
        payload["physical_time_" + tag] = c.TAU / Gamma
        print(f"done Gamma={Gamma:g} (protocol {pid})", flush=True)

    np.savez_compressed(HERE / "supp_S4_rate_scale_data_U8000.npz", **payload)
    pd.DataFrame(rows).to_csv(HERE / "supp_S4_rate_scale_protocol_errors_U8000.csv", index=False)
    (HERE / "supp_S4_high_depth_check_U8000.json").write_text(
        f'{{"R": {R}, "c": {C_VALUE}, "max_abs_difference_K40_vs_K80": {max_check_diff}}}\n'
    )
    print(pd.DataFrame(rows)[["Gamma", "beta", "gamma", "gamma_c", "p_f", "E_trajectory"]])


if __name__ == "__main__":
    main()
