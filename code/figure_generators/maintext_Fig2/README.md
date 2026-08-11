> **This pipeline produces `Fig3.pdf`, which appears in the paper as main-text Figure 2.**
> The output filename keeps the original numbering; the folder name gives
> the destination. See the repository README for the full mapping.

# Figure 3 -- constant-pool ABM vs. selected QSS branch

Two-step, standalone pipeline; no external inputs beyond this folder.

1. `python 01_generate_data.py --workers 8`
   Solves the implicit modified-Bessel QSS fixed point (Eq. 4.8) at every
   (R, c) cell, cross-checks it with independent K=40/K=80 continued-fraction
   evaluations, simulates the exact event-driven constant-pool active
   transmission forest (R in {1,1.5,2,4}, c in {0,0.25,0.5,0.75,1}, 120
   realizations per cell, tau up to 3.25), bootstraps 95% confidence
   intervals, checks that the averaging window 3<=tau<=3.25 is at
   equilibrium, and writes every numeric result to `data/`. Pure data
   generation -- no plotting code, no matplotlib import.
   Use `--smoke` for a fast 4-replicate check written to `data/smoke/`.

2. `python 02_make_figure.py`
   Reads only the cached files in `data/` (`panel_summary.csv`,
   `qss_targets.csv`, `trajectories.npz`) and draws the six-panel figure.
   The PDF is written directly to `../../figures/Fig3.pdf`, which is the
   exact path `manuscript.tex` includes
   (`\includegraphics{figures/Fig3.pdf}`) -- re-running this script is the
   only step needed to refresh the manuscript figure. Never re-simulates.
   A PNG preview and the caption text are also written locally.

```powershell
pip install -r requirements.txt
python 01_generate_data.py --workers 8
python 02_make_figure.py
```

Simulation and bootstrap seeds are fixed (see `01_generate_data.py`), so
re-running step 1 reproduces the same cached numbers bit-for-bit.
