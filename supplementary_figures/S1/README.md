# Supplementary Figure S1 - convergence with population scale and replicate count

Reproduces the ESM_1.tex OR1-S2/OR1-S4/OR1-S5 tables and Fig. S1: the full
195-protocol crossed design (R in {2,4,6}, c in {0,0.25,0.5,0.75,1}, Gamma
in {1/4,1/2,1,2,4}, three detection-tracing decompositions collapsing to one
at c=1), simulated with 30 independent 120-realization pools at U0=8000
(702,000 trajectories) and one 120-realization pool per protocol at U0 in
{500,1000,2000} (70,200 more trajectories), master seed 20260804.

Run, in order:

    python ../run_full_design.py --threads 8   # ~20-30 min; writes ../full_design_data/
    python generate_data_S1.py
    python generate_figure_S1.py

`../run_full_design.py` is the actual simulation (shared by S1-S4; see
`../README.md`) -- it uses the numba engine in `../trajectory_core_numba.py`
because the pure-Python engine (`u8000_common.py`) would take on the order
of half a day at this scale. `generate_data_S1.py` reads
`../full_design_data/error_estimates_U8000.csv.gz` and
`error_estimates_population_scale.csv` and computes the population-size
convergence, replicate-convergence, and grid-cell-breakdown tables (and the
descriptive log-log slopes) quoted in ESM_1.tex, with no further
simulation. `generate_figure_S1.py` reads those tables and produces the
publication PDF/PNG/TIFF and frozen caption.

As a sanity check, the high-depth-reference K=40-vs-K=80 discrepancy this
run produces (`../full_design_data/hd_reference_checks.csv`) matches
ESM_1.tex's stated `2.99e-10` to 3 significant figures despite using an
independently re-simulated ABM ensemble -- the high-depth solver itself is
deterministic (scipy BDF) and depends only on (R, c), not on any Monte Carlo
draw.
