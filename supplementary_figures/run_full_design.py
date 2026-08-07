#!/usr/bin/env python3
"""Production run of the full OR1-S2 crossed design (ESM_1.tex, Table OR1.1).

195 raw-rate protocols (R in {2,4,6}, c in {0,0.25,0.5,0.75,1}, Gamma in
{1/4,1/2,1,2,4}, three detection-tracing decompositions collapsing to one at
c=1), each simulated with:

- U0=8000: 30 independent 120-realization pools (702,000 trajectories).
  Within each pool, one random permutation of the 120 trajectories is drawn
  once; the first n trajectories for n in {1,2,5,10,20,40,80,120} give
  paired ensemble-mean-trajectory-error estimates against the K=40 (checked
  against K=80) high-depth deterministic reference -- 195*30=5850 estimates
  per n, 46,800 rows total.
- U0 in {500,1000,2000} (I0=0.02*U0): 1 pool of 120 each, n=120 only
  (70,200 trajectories) -- population-size convergence.

Master seed 20260804 (ESM_1.tex Table OR1.2 "Global numerical settings").
Uses the numba engine in trajectory_core_numba.py (verified against the
pure-Python engine in generate_abm_mean_trajectory_error_convergence.py:
same E_mean, order 1e-3, for a matched protocol at c=0.5 -- see
supplementary_figures/README.md).

Writes every numeric result to full_design_data/; no plotting code. S1's
generate_data_S1.py reads the two error-estimate tables to build the
population-size, replicate-convergence, and grid-cell-breakdown summaries
quoted in ESM_1.tex. S2, S3, and S4's generate_data scripts read
pool0_selected_trajectories.npz for their specific representative
protocols (all drawn from pool 0, as ESM_1.tex states), so nothing is
resimulated twice.

Run:
    python run_full_design.py --threads 8
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import set_num_threads

HERE = Path(__file__).resolve().parent
OUT = HERE / "full_design_data"
sys.path.insert(0, str(HERE))
import trajectory_core_numba as tcn  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "abm_validation_u8000_core", HERE / "generate_abm_mean_trajectory_error_convergence.py"
)
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)

MASTER_SEED = 20260804
TAU = np.linspace(0.0, tcn.TAU_END, tcn.NUM_TIMES)
SWITCH_TAU = tcn.SWITCH_TAU
HIGH_K, CHECK_K = 40, 80
N_POOLS_U8000 = 30
REPLICATE_SIZES = (1, 2, 5, 10, 20, 40, 80, 120)
POPULATION_SCALES = (500, 1000, 2000)
I0_FRACTION = 0.02

# Representative protocols reused (pool 0 only) by S2, S3, and S4 -- all at
# U0=8000, R=4, matching the specific rows ESM_1.tex figures S2-S4 display.
NEEDED_SELECTORS = (
    # S2: canonical decomposition (F, or C at c=1) at Gamma=1, all c.
    dict(R=4.0, Gamma=1.0, c=0.0, decomposition_id="F"),
    dict(R=4.0, Gamma=1.0, c=0.25, decomposition_id="F"),
    dict(R=4.0, Gamma=1.0, c=0.5, decomposition_id="F"),
    dict(R=4.0, Gamma=1.0, c=0.75, decomposition_id="F"),
    dict(R=4.0, Gamma=1.0, c=1.0, decomposition_id="C"),
    # S3: matched decomposition comparison at R=4, c=0.5, Gamma=1.
    dict(R=4.0, Gamma=1.0, c=0.5, decomposition_id="L"),
    # S4: rate-scale collapse, F decomposition, R=4, c=0.5, Gamma varying.
    dict(R=4.0, Gamma=0.25, c=0.5, decomposition_id="F"),
    dict(R=4.0, Gamma=0.5, c=0.5, decomposition_id="F"),
    dict(R=4.0, Gamma=2.0, c=0.5, decomposition_id="F"),
    dict(R=4.0, Gamma=4.0, c=0.5, decomposition_id="F"),
)


def matches(protocol: tcn.Protocol, selector: dict) -> bool:
    return (
        protocol.R == selector["R"] and protocol.Gamma == selector["Gamma"]
        and protocol.c == selector["c"] and protocol.decomposition_id == selector["decomposition_id"]
    )


def build_hd_references() -> dict[tuple[float, float], dict[str, np.ndarray]]:
    refs = {}
    for R in tcn.R_GRID:
        for c in tcn.C_GRID:
            accepted = _core.solve_high_depth(R, c, TAU, SWITCH_TAU, HIGH_K, I0_FRACTION)
            deeper = _core.solve_high_depth(R, c, TAU, SWITCH_TAU, CHECK_K, I0_FRACTION)
            refs[(R, c)] = {
                "accepted": accepted[:, :3],
                "max_abs_diff": float(np.max(np.abs(accepted[:, :3] - deeper[:, :3]))),
            }
    return refs


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="2 protocols, 2 pools, for a fast end-to-end check")
    args = ap.parse_args()
    set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    global N_POOLS_U8000
    if args.smoke:
        N_POOLS_U8000 = 2

    started = time.perf_counter()
    print("Building high-depth deterministic references (15 cells)...", flush=True)
    hd_refs = build_hd_references()
    with (OUT / "hd_reference_checks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["R", "c", "high_depth_order", "check_depth_order", "max_abs_difference_U_I_M1"])
        for (R, c), info in hd_refs.items():
            writer.writerow([R, c, HIGH_K, CHECK_K, info["max_abs_diff"]])
    print(f"  max abs difference across all cells: {max(v['max_abs_diff'] for v in hd_refs.values()):.3e}", flush=True)

    protocol_cache: dict[int, list[tcn.Protocol]] = {}
    for U0 in (8000,) + POPULATION_SCALES:
        I0 = int(round(I0_FRACTION * U0))
        protocol_cache[U0] = tcn.build_protocols(U0, I0)
    with (OUT / "protocol_table_U8000.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["protocol_id", "R", "c", "Gamma", "decomposition_id", "decomposition", "beta", "gamma", "gamma_c", "p_f", "U0", "I0"])
        for p in protocol_cache[8000]:
            writer.writerow([p.protocol_id, p.R, p.c, p.Gamma, p.decomposition_id, p.decomposition, p.beta, p.gamma, p.gamma_c, p.p_f, p.U0, p.I0])

    pool0_selected: dict[str, np.ndarray] = {}
    n_selected = 0

    replicate_rows_path = OUT / "error_estimates_U8000.csv.gz"
    replicate_fields = ["protocol_id", "R", "c", "Gamma", "decomposition_id", "pool", "n_replicates", "E_U", "E_I", "E_M1", "E_mean"]
    replicate_handle = gzip.open(replicate_rows_path, "wt", newline="", encoding="utf-8")
    replicate_writer = csv.DictWriter(replicate_handle, fieldnames=replicate_fields)
    replicate_writer.writeheader()

    protocols_8000 = protocol_cache[8000]
    if args.smoke:
        needed_ids = {p.protocol_id for sel in NEEDED_SELECTORS for p in protocols_8000 if matches(p, sel)}
        protocols_8000 = [p for p in protocols_8000 if p.protocol_id in needed_ids][:6]
    print(f"Simulating U0=8000: {len(protocols_8000)} protocols x {N_POOLS_U8000} pools x 120 replicates...", flush=True)
    for pi, protocol in enumerate(protocols_8000):
        ref = hd_refs[(protocol.R, protocol.c)]["accepted"]
        for pool in range(N_POOLS_U8000):
            seed = tcn.pool_seed(MASTER_SEED, protocol, pool)
            raw = tcn.simulate_pool(protocol, 120, TAU, SWITCH_TAU, seed)  # (120, 151, 3)

            if pool == 0:
                for selector in NEEDED_SELECTORS:
                    if matches(protocol, selector):
                        pool0_selected[protocol.protocol_id] = raw.copy()
                        n_selected += 1
                        break

            perm_rng = np.random.default_rng(seed.spawn(1)[0])
            order = perm_rng.permutation(120)
            raw_permuted = raw[order]
            raw_z = raw_permuted / protocol.U0
            for n in REPLICATE_SIZES:
                mean = raw_z[:n].mean(axis=0)
                comp = _core.integrated_rmse(mean[None, :, :], ref, TAU)[0]
                replicate_writer.writerow({
                    "protocol_id": protocol.protocol_id, "R": protocol.R, "c": protocol.c,
                    "Gamma": protocol.Gamma, "decomposition_id": protocol.decomposition_id,
                    "pool": pool, "n_replicates": n,
                    "E_U": float(comp[0]), "E_I": float(comp[1]), "E_M1": float(comp[2]),
                    "E_mean": float(np.linalg.norm(comp)),
                })
        if (pi + 1) % 20 == 0 or pi + 1 == len(protocols_8000):
            elapsed = time.perf_counter() - started
            print(f"  [{pi+1}/{len(protocols_8000)}] protocols done, elapsed {elapsed:.0f}s", flush=True)
    replicate_handle.close()
    print(f"Selected {n_selected}/{len(NEEDED_SELECTORS)} pool-0 protocols for S2-S4.", flush=True)

    np.savez_compressed(OUT / "pool0_selected_trajectories.npz", **pool0_selected)

    population_rows = []
    for U0 in POPULATION_SCALES:
        protocols = protocol_cache[U0]
        if args.smoke:
            protocols = protocols[:6]
        for protocol in protocols:
            ref = hd_refs[(protocol.R, protocol.c)]["accepted"]
            seed = tcn.pool_seed(MASTER_SEED, protocol, 0)
            raw = tcn.simulate_pool(protocol, 120, TAU, SWITCH_TAU, seed)
            raw_z = raw / protocol.U0
            mean = raw_z.mean(axis=0)
            comp = _core.integrated_rmse(mean[None, :, :], ref, TAU)[0]
            population_rows.append({
                "U0": U0, "I0": protocol.I0, "protocol_id": protocol.protocol_id,
                "R": protocol.R, "c": protocol.c, "n_replicates": 120,
                "E_U": float(comp[0]), "E_I": float(comp[1]), "E_M1": float(comp[2]),
                "E_mean": float(np.linalg.norm(comp)),
            })
        print(f"U0={U0} population-scale pools done ({len(protocols)} protocols).", flush=True)

    with (OUT / "error_estimates_population_scale.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(population_rows[0].keys()))
        writer.writeheader()
        writer.writerows(population_rows)

    elapsed = time.perf_counter() - started
    manifest = {
        "design": "ESM_1.tex OR1-S2 crossed design (195 raw-rate protocols)",
        "master_seed": MASTER_SEED,
        "U0_grid": [500, 1000, 2000, 8000],
        "I0_fraction": I0_FRACTION,
        "R_grid": list(tcn.R_GRID), "c_grid": list(tcn.C_GRID), "Gamma_grid": list(tcn.GAMMA_GRID),
        "pools_at_U0_8000": N_POOLS_U8000,
        "replicate_sizes": list(REPLICATE_SIZES),
        "high_depth_order": HIGH_K, "check_depth_order": CHECK_K,
        "total_U0_8000_trajectories": len(protocols_8000) * N_POOLS_U8000 * 120,
        "total_population_scale_trajectories": len(POPULATION_SCALES) * len(protocols_8000) * 120,
        "elapsed_seconds": elapsed,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Full design complete in {elapsed/60:.1f} minutes.", flush=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
