> **This pipeline produces `Fig5.png`, which appears in the paper as main-text Figure 3.**
> The output filename keeps the original numbering; the folder name gives
> the destination. See the repository README for the full mapping.

# Figure 5 -- analytical growth/replacement curves and ABM validation

Two-step, standalone pipeline. Uses `canonical_qss.py` (the same shared,
verified solver as Figure 4) and `figure3_measurement_windows.json` (Figure 3's
per-(R,c)-cell equilibration protocol -- a static input, not regenerated here;
it only supplies the entry-window start time for each cohort simulation).

1. `python 01_generate_data.py`
   Computes the canonical growth rate g(R,c), incident-cohort lifetime
   reproduction R_time(R,c), and the actual control boundary c*(R) up to its
   c=1 endpoint (deterministic, via `canonical_qss.py`). Then independently
   validates those curves with a constant-pool one-step forward-tracing ABM
   (an event-driven active-forest simulation, unrelated to the QSS solver):
   CP_COHORT_R2 (complete-follow-up lifetime estimates on the same (R,c) grid)
   and CP_THRESHOLD_R1 (growth estimates bracketing the control boundary).
   Bootstraps an ABM-inferred critical R (95% CI) at each traced c. Pure data
   generation -- no plotting code, no matplotlib import. Writes every numeric
   result to `data/`. Use `--smoke` for a fast check with small replicate
   counts and coarse grids; use `--workers N` to parallelize the ABM paths.

2. `python 02_make_figure.py`
   Reads only the cached files in `data/` and draws the three-panel figure
   (Malthusian growth, incident-cohort replacement, actual control boundary).
   The PNG is written directly to `../../figures/Fig5.png`, which is the exact
   path `manuscript.tex` includes (`\includegraphics{figures/Fig5.png}`) --
   re-running this script is the only step needed to refresh the manuscript
   figure. Never re-solves or re-simulates anything.

```powershell
pip install -r requirements.txt
python 01_generate_data.py --workers 8
python 02_make_figure.py
```

Note: the ABM stage runs ~9,200 stochastic paths (production replicate
counts), so the full run takes noticeably longer than Figures 1-4; use
`--smoke` first to sanity-check the pipeline.
