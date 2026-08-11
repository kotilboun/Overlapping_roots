#!/usr/bin/env python3
"""Step 1/2 for Figure 9: representative R=4 trajectories vs. the algebraic QSS closure.

Simulates the finite-susceptible-pool event-driven ABM at R=4 for
c in {0, 0.25, 0.5, 0.75, 1} (same engine and master seed 20260815 as
Figure 6/Figure 7, via ``shared_trajectory_engine/trajectory_core.py``), solves the
zeroth-order algebraic QSS closure (Eq. 7.6), and writes:

- pointwise ABM ensemble-mean and 2.5th-97.5th replicate-percentile bands
  for i=I/U0, u=U/U0, m1=M1/I, and flux=c*m1*i (what Figure 9 plots);
- the same four algebraic-closure trajectories;
- decomposed trajectory-error metrics (closure vs. ABM mean, Eq. 7.16, with
  95% Monte Carlo intervals) and a stochastic-spread metric (replicate
  spread around the ABM mean), each over the full trajectory and the
  post-activation window tau>=0.5.

This folder duplicates Figure 8's ABM simulation (identical engine and seed)
so it remains self-contained; only the overlaid closure differs. Pure data
generation -- no matplotlib import. Run:
    python 01_generate_data.py --workers 8

A fast end-to-end check is available with:
    python 01_generate_data.py --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SHARED = HERE.parents[0] / "shared_trajectory_engine"
sys.path.insert(0, str(SHARED))

import trajectory_core as core  # noqa: E402

R_GRID_FOR_SEEDING = (1.0, 1.5, 2.0, 4.0)
R_VALUE = 4.0
C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
VARIABLES = ("i", "u", "m1", "flux")


def archive_key(c: float) -> str:
    return f"c{c:g}".replace(".", "p")


def _simulate_cell(args: tuple[float, core.Protocol, np.random.SeedSequence]) -> tuple[float, np.ndarray]:
    c, protocol, seed = args
    return c, core.simulate_replicates(R_VALUE, c, protocol, seed)


def simulate_all(protocol: core.Protocol, workers: int) -> dict[float, np.ndarray]:
    tasks = [
        (c, protocol, core.cell_seed(protocol.seed, R_GRID_FOR_SEEDING, C_VALUES, R_VALUE, c))
        for c in C_VALUES
    ]
    if workers <= 1:
        results = [_simulate_cell(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_simulate_cell, task): task[0] for task in tasks}
            for done, future in enumerate(as_completed(futures), start=1):
                c, raw = future.result()
                results.append((c, raw))
                print(f"[{done}/{len(tasks)}] c={c:g}", flush=True)
    return dict(results)


def algebraic_closure_state(R: float, c: float, times: np.ndarray, U0: float, I0: float, tau_on: float, max_step: float) -> np.ndarray:
    """Columns (U, I, m1) for the zeroth-order algebraic closure (m1 derived)."""
    state = core.solve_algebraic(R, c, times, max_step, U0, I0, tau_on)
    U, I = state[:, 0], state[:, 1]
    m1 = core.m10(R * U / U0)
    return np.column_stack((U, I, m1))


def closure_z(state: np.ndarray, U0: float) -> np.ndarray:
    """z=(U/U0, I/U0, M1/U0) from a closure state with columns (U, I, m1)."""
    U, I, m1 = state[:, 0], state[:, 1], state[:, 2]
    return np.column_stack((U / U0, I / U0, I * m1 / U0))


def decomposed_metrics_rows(c: float, times: np.ndarray, raw_z: np.ndarray, closures: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    masks = {"full_trajectory": np.ones(len(times), dtype=bool), "post_activation": times >= 0.5}
    for interval, mask in masks.items():
        sub_times = times[mask]
        sub_raw_z = raw_z[:, mask, :]
        spread = core.stochastic_spread(sub_raw_z, sub_times)
        rows.append({
            "c": c, "interval": interval, "layer": "stochastic_spread_around_ABM_mean",
            "model": "ABM_replicates", "error": spread, "standard_error": None, "ci_low": None, "ci_high": None,
        })
        for name, state_z in closures.items():
            point, se, low, high = core.delta_method_interval(sub_raw_z, state_z[mask], sub_times)
            rows.append({
                "c": c, "interval": interval, "layer": "closure_minus_ABM_mean",
                "model": name, "error": point, "standard_error": se, "ci_low": low, "ci_high": high,
            })
    return rows


def verify_against_figure6(abm_by_c: dict[float, np.ndarray]) -> bool:
    fig6_archive = HERE.parents[0] / "maintext_Fig4" / "data" / "source_abm_trajectories.npz"
    if not fig6_archive.exists():
        return False
    with np.load(fig6_archive) as archive:
        for c in C_VALUES:
            key = f"R{R_VALUE:g}_c{c:g}".replace(".", "p")
            if key not in archive:
                return False
            if not np.array_equal(archive[key], abm_by_c[c]):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    protocol = core.Protocol()
    if args.smoke:
        protocol = core.Protocol(replicates=4, U0=400, I0=20, tau_end=1.2, num_times=25)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    abm_by_c = simulate_all(protocol, args.workers)
    times = np.linspace(0.0, protocol.tau_end, protocol.num_times)
    U0, I0, tau_on = float(protocol.U0), float(protocol.I0), protocol.tau_on

    bitwise_match_figure6 = verify_against_figure6(abm_by_c) if not args.smoke else None

    summary_payload: dict[str, np.ndarray] = {"times": times, "c_values": np.asarray(C_VALUES)}
    closure_payload: dict[str, np.ndarray] = {"times": times, "c_values": np.asarray(C_VALUES)}
    metric_rows: list[dict[str, object]] = []

    for c in C_VALUES:
        raw = abm_by_c[c]
        raw_z = raw[:, :, :3] / U0
        derived = core.i_u_m1_flux(raw[:, :, :3], c, U0)
        key = archive_key(c)
        for variable in VARIABLES:
            summary = core.pointwise_summary(derived[variable])
            for statistic, values in summary.items():
                summary_payload[f"abm_{variable}_{statistic}_{key}"] = values

        state = algebraic_closure_state(R_VALUE, c, times, U0, I0, tau_on, 1.0 / 60.0)
        closures_z = {"algebraic_qss0": closure_z(state, U0)}
        derived_closure = core.closure_i_u_m1_flux(state, c, U0)
        for variable in VARIABLES:
            closure_payload[f"algebraic_qss0_{variable}_{key}"] = derived_closure[variable]

        metric_rows.extend(decomposed_metrics_rows(c, times, raw_z, closures_z))

    np.savez_compressed(DATA_DIR / "abm_trajectory_summary.npz", **summary_payload)
    np.savez_compressed(DATA_DIR / "closure_trajectories.npz", **closure_payload)
    with (DATA_DIR / "decomposed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    elapsed = time.perf_counter() - started
    manifest = {
        "figure": 8,
        "status": "production" if not args.smoke else "smoke_test",
        "content": "representative R=4 finite-pool trajectories vs. the zeroth-order algebraic QSS closure",
        "protocol": asdict(protocol),
        "R": R_VALUE,
        "c_values": list(C_VALUES),
        "bitwise_identical_to_figure6_R4_cells": bitwise_match_figure6,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "elapsed_seconds": elapsed,
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Generated Figure 9 data for R={R_VALUE:g}, {len(C_VALUES)} c-values, "
        f"bitwise match to Figure 7 R=4 cells: {bitwise_match_figure6}, elapsed {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
