# Supplementary ABM-validation figures (Online Resource 1 / ESM_1.tex)

Two production designs live in this folder:

**S1-S4** reproduce the full crossed design ESM_1.tex documents (sections
OR1-S1--OR1-S5): 195 raw-rate protocols (R in {2,4,6}, c in
{0,0.25,0.5,0.75,1}, Gamma in {1/4,1/2,1,2,4}, three detection-tracing
decompositions collapsing to one at c=1), 30 independent 120-realization
pools at U0=8000 (702,000 trajectories) plus one 120-realization pool per
protocol at U0 in {500,1000,2000} (70,200 more trajectories), compared
against a high-depth deterministic reference (K=40, checked against K=80),
master seed 20260804. `run_full_design.py` runs this whole design once
(shared by S1-S4, ~20-30 minutes with the numba engine in
`trajectory_core_numba.py`); each `SN/generate_data_SN.py` then reads the
resulting tables/pool-0 trajectories with no further simulation.

**S5** is the separate OR1-S6 direct event-level verification: the same
195-protocol grid at fixed U0=8000, but simulated to directly count
transmission/removal/identification/traced-removal events and compare them
against the ABM-supplied analytical intensities, rather than comparing
trajectories to a deterministic reference. It has its own numba engine
(`S5/s5_common.py`) and master seed (20260817).

Selected production setting for S1-S4:
- U0 = 8000 (S1-S4), I0 = 0.02*U0
- 120 independent ABM realizations per pool; 30 pools at U0=8000
- high-depth deterministic reference K = 40, checked against K = 80
- tracing activation at tau = 0.5
- master seed 20260804

Files:
- Supplementary Figure S1: convergence with population scale and replicate count.
- Supplementary Figure S2: selected U0=8000 ABM means versus the high-depth hierarchy across c.
- Supplementary Figure S3: matched-c raw-rate decomposition comparison at R=4 and c=0.5.
- Supplementary Figure S4: rate-scale collapse check.
- Supplementary Figure S5: direct event-term verification (see S5/README.md).

## Layout

- `run_full_design.py` -- the S1-S4 production simulation. Run once:
  `python run_full_design.py --threads 8`. Writes `full_design_data/`
  (error-estimate tables, high-depth reference checks, and the pool-0
  trajectories needed by S2-S4's representative figures).
- `trajectory_core_numba.py` -- the numba-accelerated ABM engine and
  195-protocol design shared by `run_full_design.py`.
- `S1/` -- `generate_data_S1.py` then `generate_figure_S1.py`: the
  population-size/replicate-convergence/grid-cell-breakdown tables and Fig. S1.
- `S2/` -- `generate_data_S2.py` then `generate_figure_S2.py`: canonical
  decomposition at R=4, Gamma=1, all c.
- `S3/` -- `generate_data_S3.py` then `generate_figure_S3.py`: matched
  decomposition comparison at R=4, c=0.5.
- `S4/` -- `generate_data_S4.py` then `generate_figure_S4.py`: rate-scale
  collapse at R=4, c=0.5, Gamma varying.
- `S5/` -- `generate_data_S5.py` then `generate_figure_S5.py`, sharing the
  model core in `S5/s5_common.py` (needs `numba`, see `S5/requirements.txt`).

S1-S4 also share the pure-Python deterministic-reference solver at this
folder's top level (`generate_abm_mean_trajectory_error_convergence.py`,
via `u8000_common.py`) -- only the stochastic ABM simulation itself needed
the numba engine for speed; the high-depth reference (scipy BDF) is cheap
regardless (15 (R,c) cells total).

## Reproducing from scratch

```
python run_full_design.py --threads 8
```

then, from each of `S1/`, `S2/`, `S3/`, `S4/`:

```
python generate_data_SN.py
python generate_figure_SN.py
```

`S5/` is independent and self-contained (`python generate_data_S5.py
--threads 5` then `python generate_figure_S5.py`).
