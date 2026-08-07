# Figure 7 -- closure-trajectory validation against the ensemble-mean ABM

Two-step, standalone pipeline; no external inputs beyond this folder. Uses
the identical finite-susceptible-pool event-driven ABM engine as Figure 6
(same engine, same master seed 20260815), re-run here rather than shared, so
this folder does not depend on Figure 6's cached data.

## Frozen display protocol

- $R=\{1,1.5,2,4\}$
- $c=\{0,0.25,0.5,0.75,1\}$
- $S_0=8000$, $I_0=160$
- 120 independent ABM realizations per $(R,c)$ cell
- tracing activation at $\tau=0.5$
- 151 observation times on $0\leq\tau\leq5$

1. `python 01_generate_data.py --workers 8`
   Runs the finite-susceptible-pool event-driven ABM, solves the zeroth-order
   algebraic QSS closure and the $K=1,2,3$ dynamic tail closures with a
   switch-restarted ODE integrator, and computes the normalized integrated
   $L^2$ trajectory error $E_C$ (Eq. 7.16, formerly Eq. 7.19 in the archived
   numbering) between each closure and the ensemble-mean ABM trajectory, with
   whole-realization influence-function/delta-method 95% Monte Carlo
   intervals. Pure data generation -- no matplotlib import. Writes every
   numeric result to `data/`. Use `--smoke` for a fast end-to-end check with
   small replicate counts and a short time horizon.

2. `python 02_make_figure.py`
   Reads only the cached `data/trajectory_error_summary.csv` and draws the
   four-panel log-scale trajectory-error figure. The PDF is written directly
   to `../../figures/Fig7.pdf`, the exact path `sn-article.tex` includes
   (`\includegraphics{figures/Fig7.pdf}`) -- re-running this script is the
   only step needed to refresh the manuscript figure. Never re-simulates the
   ABM or re-solves the closures.

```powershell
pip install -r requirements.txt
python 01_generate_data.py --workers 8
python 02_make_figure.py
```

## Relationship to Figures 6, 8, and 9

Figure 6 evaluates the closures snapshotwise, at the ensemble-mean ABM
state, at each observation time. Figure 7 instead integrates the closure
trajectory error $E_C$ over the whole trajectory and reports it across the
full $(R,c)$ grid. Figures 8 and 9 show what that error looks like as an
actual trajectory overlay at the representative case $R=4$ -- for the
dynamic closures (Fig. 8) and the algebraic QSS closure (Fig. 9)
respectively -- reusing this folder's R=4 ABM replicates bit-for-bit
(verified in `Figure_8/01_generate_data.py` and `Figure_9/01_generate_data.py`).
