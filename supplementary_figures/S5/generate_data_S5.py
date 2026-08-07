#!/usr/bin/env python3
"""Data generator for Supplementary Fig. S5: direct ABM event-term verification.

The instrumented event-driven ABM counts transmission, spontaneous removal,
identification, and traced-removal contributions. Over the same paths it
integrates the ABM-supplied state variables entering the analytical
intensities beta S I, gamma I, gamma_c I, and gamma_c p_f M1.

Default production design: U0=8000, I0=160, 195 crossed raw-rate protocols,
120 independent realizations per protocol, and 150 intervals on 0 <= tau <= 5.

This produces only the numerical data (point archive, protocol table, panel
summary statistics, compact NPZ arrays, metadata) -- run
generate_figure_S5.py afterward to render the publication figure and caption
from these saved files without resimulating.

Run:
    python generate_data_S5.py --threads 5
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bmb-event-term-mpl"))

import numpy as np
from numba import set_num_threads

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s5_common as s5  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=HERE)
    ap.add_argument("--replicates", type=int, default=s5.N_REPLICATES)
    ap.add_argument("--threads", type=int, default=min(5, os.cpu_count() or 2))
    ap.add_argument("--master-seed", type=int, default=s5.MASTER_SEED)
    args = ap.parse_args()
    set_num_threads(args.threads)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocols = s5.build_protocols()
    # Trigger JIT with a minimal run.
    _ = s5.simulate_batch(s5.U0, s5.I0, 4/8000, 0.5, 0.5, 1.0, 1.0, 0.1, 3, 0.05, np.array([np.uint64(123)], dtype=np.uint64))

    all_rows = []
    protocol_rows = [asdict(p) for p in protocols]
    observed_cube = np.zeros((len(protocols), s5.N_BINS, 4), float)
    expected_cube = np.zeros_like(observed_cube)
    exposure_cube = np.zeros((len(protocols), s5.N_BINS, 3), float)
    for pi, p in enumerate(protocols):
        seeds = np.array([s5.trajectory_seed(p, r, args.master_seed) for r in range(args.replicates)], dtype=np.uint64)
        result = s5.simulate_batch(s5.U0, s5.I0, p.beta, p.gamma, p.gamma_c, p.p_f, p.Gamma,
                                    s5.TAU_END, s5.N_BINS, s5.TRACE_ON, seeds)
        observed = result[:, :, :4].mean(axis=0)
        exposures = result[:, :, 4:].mean(axis=0)
        expected = s5.analytical_expected(p, exposures)
        observed_cube[pi] = observed; expected_cube[pi] = expected; exposure_cube[pi] = exposures
        dtau = s5.TAU_END / s5.N_BINS
        for j in range(s5.N_BINS):
            row = {
                "protocol_id": p.protocol_id, "R": p.R, "c": p.c, "Gamma": p.Gamma,
                "decomposition_id": p.decomposition_id,
                "tau_start": j*dtau, "tau_end": (j+1)*dtau,
                "mean_integral_UI_dtau": exposures[j, 0],
                "mean_integral_I_dtau": exposures[j, 1],
                "mean_integral_active_M1_dtau": exposures[j, 2],
            }
            for k, event in enumerate(s5.EVENT_NAMES):
                row[f"expected_{event}"] = expected[j, k]
                row[f"observed_{event}"] = observed[j, k]
            all_rows.append(row)
        if (pi+1) % 10 == 0 or pi+1 == len(protocols):
            print(f"completed {pi+1}/{len(protocols)} protocols", flush=True)

    zero_rows = [r for r in all_rows if r["expected_traced_removal"] == 0.0]
    zero_violations = int(sum(bool(r["observed_traced_removal"] != 0.0) for r in zero_rows))
    if zero_violations:
        raise AssertionError(f"{zero_violations} traced removals occurred under zero analytical tracing intensity")

    summaries = s5.compute_panel_summaries(all_rows)

    with gzip.open(out/"supp_S5_event_term_validation_points.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    with open(out/"supp_S5_event_term_validation_protocols.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(protocol_rows[0])); w.writeheader(); w.writerows(protocol_rows)
    with open(out/"supp_S5_event_term_validation_panel_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    np.savez_compressed(out/"supp_S5_event_term_validation_ensemble_data.npz",
                        tau_edges=np.linspace(0, s5.TAU_END, s5.N_BINS+1),
                        observed_mean_counts=observed_cube,
                        analytical_mean_expected_counts=expected_cube,
                        mean_state_exposures=exposure_cube,
                        protocol_ids=np.array([p.protocol_id for p in protocols]),
                        event_names=np.array(s5.EVENT_NAMES))
    meta = {
        "figure": "Supplementary Figure S5",
        "purpose": "direct event-count versus ABM-supplied analytical-term comparison",
        "U0": s5.U0, "I0": s5.I0, "R_grid": list(s5.R_GRID), "c_grid": list(s5.C_GRID), "Gamma_grid": list(s5.GAMMA_GRID),
        "raw_rate_protocols": len(protocols), "replicates_per_protocol": args.replicates,
        "observation_intervals": s5.N_BINS, "protocol_interval_points": len(all_rows),
        "trace_activation_tau": s5.TRACE_ON, "master_seed": args.master_seed,
        "zero_expected_trace_points": len(zero_rows),
        "zero_expected_trace_points_with_observed_trace_removals": zero_violations,
        "panel_metrics": summaries,
        "analytical_terms": {"transmission": "beta S I", "recovery": "gamma I", "identification": "gamma_c I", "traced_removal": "gamma_c p_f M1"},
    }
    (out/"supp_S5_event_term_validation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
