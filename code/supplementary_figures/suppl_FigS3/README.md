> **This pipeline produces `supp_S2_selected_production_validation_U8000.pdf`, which appears in the paper as Supplementary Material 1, Fig. S3.**
> The output filename keeps the original numbering; the folder name gives
> the destination. See the repository README for the full mapping.

# Supplementary Figure S2 - selected production validation at U0=8000

The canonical decomposition (Gamma=1, gamma=0, gamma_c=1, p_f=c) at R=4 for
c in {0,0.25,0.5,0.75,1}, 120 ABM realizations per c-value, compared against
the high-depth deterministic reference (K=40, checked against K=80).

Run, in order (after `python ../run_full_design.py --threads 8` has
populated `../full_design_data/`):

    python generate_data_S2.py
    python generate_figure_S2.py

`generate_data_S2.py` reads these 5 protocols' pool-0 replicate trajectories
from `../full_design_data/pool0_selected_trajectories.npz` -- they are part
of the same 195-protocol, master-seed-20260804 crossed design used by
S1, S3, and S4 (`../run_full_design.py`; see `../README.md`), not a
separate simulation. `generate_figure_S2.py` reads the resulting data files
and produces the publication PDF/PNG/TIFF and frozen caption without
resimulating.
