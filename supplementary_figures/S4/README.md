# Supplementary Figure S4 - rate-scale collapse

R=4, c=0.5, Gamma in {0.25, 0.5, 1, 2, 4}, frequent-detection/partial-tracing
decomposition (gamma=0, gamma_c=Gamma, p_f=c), 120 ABM realizations per
Gamma, compared against the common high-depth deterministic reference for
(R=4, c=0.5).

Run, in order (after `python ../run_full_design.py --threads 8` has
populated `../full_design_data/`):

    python generate_data_S4.py
    python generate_figure_S4.py

This figure's five protocols are pool 0 of the same 195-protocol,
master-seed-20260804 crossed design (`../run_full_design.py`) used by S1-S3
-- see `../README.md`. No separate simulation is run here.
