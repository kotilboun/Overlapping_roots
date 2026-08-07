# Supplementary Figure S3 - matched raw-rate decomposition comparison

The two matched raw-rate protocols at R=4, c=0.5, U0=8000 (frequent
detection with partial tracing, and less-frequent detection with complete
tracing; both share Gamma=1 and therefore the same nondimensional
deterministic reference), 120 ABM realizations each.

Run, in order (after `python ../run_full_design.py --threads 8` has
populated `../full_design_data/`):

    python generate_data_S3.py
    python generate_figure_S3.py

`generate_data_S3.py` reads these 2 protocols' pool-0 replicate trajectories
from `../full_design_data/pool0_selected_trajectories.npz` -- they are part
of the same 195-protocol, master-seed-20260804 crossed design used by
S1, S2, and S4 (`../run_full_design.py`; see `../README.md`), not a
separate simulation. `generate_figure_S3.py` reads the resulting data files
and produces the publication PDF/PNG/TIFF and frozen caption without
resimulating.
