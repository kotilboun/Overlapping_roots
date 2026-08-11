#!/usr/bin/env python3
"""Data generator for Supplementary Figure S1 (convergence with population
scale and replicate count) and the supplementary_material_1.tex S5.2/S5.4/S5.5 tables.

Unlike the smaller 30-cell pilot this script previously ran, this now
reproduces the full crossed design supplementary_material_1.tex documents: 195 raw-rate
protocols x 30 independent 120-realization pools at U0=8000, plus one
120-realization pool per protocol at U0 in {500,1000,2000}. That design is
simulated once, by ../run_full_design.py (numba-accelerated; the pure-Python
engine here would take too long) -- run that first:

    python ../run_full_design.py --threads 8

This script then reads ../full_design_data/error_estimates_U8000.csv.gz and
../full_design_data/error_estimates_population_scale.csv and computes
exactly the three summary tables and the S5.5 text statistics quoted in
supplementary_material_1.tex, with no further simulation.

Run:
    python generate_data_S1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FULL_DESIGN = HERE.parent / "full_design_data"
REPLICATE_SIZES = (1, 2, 5, 10, 20, 40, 80, 120)
U0_GRID = (500, 1000, 2000, 8000)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
        "n_estimates": int(len(values)),
    }


def log_log_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Descriptive least-squares slope of log(y) on log(x)."""
    slope, _ = np.polyfit(np.log(x), np.log(y), 1)
    return float(slope)


def main() -> None:
    replicate = pd.read_csv(FULL_DESIGN / "error_estimates_U8000.csv.gz")
    population = pd.read_csv(FULL_DESIGN / "error_estimates_population_scale.csv")
    hd_checks = pd.read_csv(FULL_DESIGN / "hd_reference_checks.csv")

    # --- Population-size convergence table (one row per U0). ---------------
    pop_rows = []
    for U0 in U0_GRID:
        if U0 == 8000:
            values = replicate.loc[replicate.pool.eq(0) & replicate.n_replicates.eq(120), "E_mean"].to_numpy()
            protocols = replicate.loc[replicate.pool.eq(0) & replicate.n_replicates.eq(120), "protocol_id"].nunique()
        else:
            values = population.loc[population.U0.eq(U0), "E_mean"].to_numpy()
            protocols = population.loc[population.U0.eq(U0), "protocol_id"].nunique()
        I0 = int(round(0.02 * U0))
        stat = summarize(values)
        pop_rows.append({"U0": U0, "I0": I0, "protocols": protocols, "estimates": stat["n_estimates"],
                          "median": stat["median"], "q10": stat["q10"], "q90": stat["q90"]})
    pop_df = pd.DataFrame(pop_rows)
    pop_df.to_csv(HERE / "population_size_convergence.csv", index=False)

    # --- Replicate convergence table (one row per n, at U0=8000). ----------
    rep_rows = []
    for n in REPLICATE_SIZES:
        values = replicate.loc[replicate.n_replicates.eq(n), "E_mean"].to_numpy()
        protocols = replicate.loc[replicate.n_replicates.eq(n), "protocol_id"].nunique()
        pools = replicate.loc[replicate.n_replicates.eq(n), "pool"].nunique()
        stat = summarize(values)
        rep_rows.append({"n_replicates": n, "protocols": protocols, "pools_per_protocol": pools,
                          "estimates": stat["n_estimates"], "median": stat["median"],
                          "q10": stat["q10"], "q90": stat["q90"]})
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(HERE / "replicate_convergence_U8000.csv", index=False)

    # --- Grid-cell breakdown table (R, c at U0=8000, n=120). ---------------
    at120 = replicate.loc[replicate.n_replicates.eq(120)]
    grid_rows = []
    for (R, cv), group in at120.groupby(["R", "c"]):
        stat = summarize(group["E_mean"].to_numpy())
        grid_rows.append({
            "R": R, "c": cv, "protocols": group.protocol_id.nunique(), "estimates": stat["n_estimates"],
            "median": stat["median"], "mean": stat["mean"], "q90": stat["q90"], "max": stat["max"],
        })
    grid_df = pd.DataFrame(grid_rows).sort_values(["R", "c"])
    grid_df.to_csv(HERE / "grid_cell_breakdown_U8000.csv", index=False)

    # --- S5.5 top-level descriptive statistics. ---------------------------
    n120 = at120["E_mean"].to_numpy()
    pop_slope = log_log_slope(pop_df.U0.to_numpy(dtype=float), pop_df["median"].to_numpy())
    rep_slope = log_log_slope(rep_df.n_replicates.to_numpy(dtype=float), rep_df["median"].to_numpy())
    summary = {
        "design": "195 raw-rate protocols x 30 pools (U0=8000) + x1 pool (U0<8000), master seed 20260804",
        "n120_median": float(np.median(n120)), "n120_mean": float(np.mean(n120)),
        "n120_q90": float(np.quantile(n120, 0.9)), "n120_max": float(np.max(n120)),
        "population_size_median_U500": float(pop_df.loc[pop_df.U0 == 500, "median"].iloc[0]),
        "population_size_median_U8000": float(pop_df.loc[pop_df.U0 == 8000, "median"].iloc[0]),
        "population_size_log_log_slope": pop_slope,
        "replicate_count_log_log_slope": rep_slope,
        "high_depth_max_abs_difference": float(hd_checks.max_abs_difference_U_I_M1.max()),
    }
    (HERE / "summary_stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(pop_df)
    print(rep_df)
    print(grid_df)


if __name__ == "__main__":
    main()
