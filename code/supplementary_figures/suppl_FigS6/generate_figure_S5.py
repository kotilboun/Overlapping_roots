#!/usr/bin/env python3
"""Figure generator for Supplementary Figure S5.

Reads the data files written by generate_data_S5.py (which must be run
first) -- the gzipped point archive and the panel-summary statistics -- and
renders the publication PDF/PNG/TIFF and frozen caption. Does not run any
new simulations.

Run:
    python generate_figure_S5.py
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s5_common as s5  # noqa: E402

# csv.DictReader reads everything as strings; these are the only text
# (non-numeric) columns in each data file, so every other column is cast
# back to float/int below.
POINT_TEXT_FIELDS = {"protocol_id", "decomposition_id"}
SUMMARY_TEXT_FIELDS = {"event"}
SUMMARY_INT_FIELDS = {"n_points", "zero_hazard_points_omitted_from_plot"}


def read_rows() -> list[dict]:
    """Load the per-protocol, per-interval expected/observed event counts."""
    with gzip.open(HERE / "supp_S5_event_term_validation_points.csv.gz", "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw_row in reader:
            row = dict(raw_row)
            for name in row:
                if name not in POINT_TEXT_FIELDS:
                    row[name] = float(row[name])
            rows.append(row)
        return rows


def read_panel_summaries() -> list[dict]:
    """Load the per-event calibration statistics (slope, R^2, ...)."""
    with open(HERE / "supp_S5_event_term_validation_panel_summary.csv", newline="", encoding="utf-8") as f:
        summaries = list(csv.DictReader(f))
    for stat in summaries:
        for name, value in stat.items():
            if name in SUMMARY_TEXT_FIELDS:
                continue
            stat[name] = int(value) if name in SUMMARY_INT_FIELDS else float(value)
    return summaries


def main() -> None:
    rows = read_rows()
    summaries = read_panel_summaries()

    metadata = json.loads((HERE / "supp_S5_event_term_validation_metadata.json").read_text(encoding="utf-8"))
    replicates = metadata["replicates_per_protocol"]
    master_seed = metadata["master_seed"]

    s5.plot_panels(rows, summaries, HERE)

    metric_text = "; ".join(
        f"{m['event'].replace('_',' ')}: slope {m['slope_through_origin']:.4f}, R^2 {m['r_squared']:.4f}"
        for m in summaries
    )
    caption = (
        "Supplementary Figure S5. Direct event-level verification of the four terms entering the infectious-population balance. "
        "The instrumented Gillespie ABM records transmission events, spontaneous removals, identification events, and the number of infectious direct infectees removed by tracing. "
        "For the same realized paths, the ABM state variables $S$, $I$, and $M_1$ are integrated over each observation interval to evaluate the corresponding analytical event expectations "
        "$\\int \\beta S I\\,\\mathrm{d}t$, $\\int \\gamma I\\,\\mathrm{d}t$, $\\int \\gamma_c I\\,\\mathrm{d}t$, and $\\int \\gamma_c p_f M_1\\,\\mathrm{d}t$. "
        f"Each point represents one of 195 raw-rate protocols and one of 150 observation intervals, averaged over {replicates} independent realizations (29,250 protocol-interval points before panel-specific omission of trivial zeros). "
        "The dashed line is the identity. Intervals in which the tracing intensity is identically zero are verified in the numerical table and omitted from panel (d) to avoid overplotting at the origin. "
        f"Through-origin calibration slopes and coefficients of determination were: {metric_text}. "
        "The comparison is a conditional-intensity and implementation check based on ABM-supplied states, rather than an independent fit to the deterministic trajectory."
    )
    (HERE / "supp_S5_event_term_validation_caption.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"Figure S5 written (replicates={replicates}, master_seed={master_seed}).")


if __name__ == "__main__":
    main()
