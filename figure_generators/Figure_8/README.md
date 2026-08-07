# Figure 8 -- representative trajectories vs. depth-K dynamic closures

Two-step, standalone pipeline; no external inputs beyond this folder and the
shared engine in `../figures8_9_shared/trajectory_core.py`.

## Frozen display protocol

- $R=4$ (the representative case discussed in Sect. 7.7)
- $c=\{0,0.25,0.5,0.75,1\}$
- $S_0=8000$, $I_0=160$
- 120 independent ABM realizations per $c$
- tracing activation at $\tau=0.5$
- 151 observation times on $0\leq\tau\leq5$
- master seed 20260815 (identical engine and seed to Figures 6 and 7, so the
  R=4 replicates here are bit-for-bit identical to the R=4 cells already
  simulated for Figure 7 -- checked automatically in step 1)

1. `python 01_generate_data.py --workers 8`
   Runs the finite-susceptible-pool event-driven ABM, computes pointwise
   ensemble-mean and 2.5th-97.5th replicate-percentile bands for
   $i=I/S_0$, $s=S/S_0$, $m_1=M_1/I$, and $c\,m_1 i$, solves the depth-K=1,2,3
   dynamic tail closures (Sect. 7.1), and computes decomposed
   closure-vs-ABM-mean trajectory errors (Eq. 7.16, with 95% Monte Carlo
   intervals) and a stochastic-spread metric, both over the full trajectory
   and the post-activation window. Pure data generation -- no plotting code.
   Writes every numeric result to `data/`. Use `--smoke` for a fast
   end-to-end check with small replicate counts and a short time horizon.

2. `python 02_make_figure.py`
   Reads only the cached files in `data/` and draws the four-row,
   five-column trajectory figure. The PDF is written directly to
   `../../figures/Fig8.pdf`, the exact path `sn-article.tex` includes
   (`\includegraphics{figures/Fig8.pdf}`) -- re-running this script is the
   only step needed to refresh the manuscript figure. Never re-simulates
   the ABM or re-solves the closures.

## Relationship to Figures 6, 7, and 9

Figure 7 reports the aggregated trajectory error $E_C$ (Eq. 7.16) across the
full $(R,c)$ grid; Figure 8 shows what that error looks like as an actual
trajectory overlay, at the representative case $R=4$. Figure 9 shows the
same R=4 ABM ensemble overlaid with the zeroth-order algebraic QSS closure
instead of the dynamic closures. Both folders duplicate the ABM simulation
independently (same engine and seed) so each remains self-contained.
