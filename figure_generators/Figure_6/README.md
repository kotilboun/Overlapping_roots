# Figure 6 -- ensemble-mean local closure validation

Two-step, standalone pipeline; no external inputs beyond this folder.

## Frozen display protocol

- $R=\{1,1.5,2,4\}$
- $c=\{0,0.25,0.5,0.75,1\}$
- $S_0=8000$, $I_0=160$
- 120 independent ABM realizations per $(R,c)$ cell
- tracing activation at $\tau=0.5$
- 151 observation times on $0\leq\tau\leq5$

For each parameter cell and time, the ABM retained variables are averaged
over the ensemble first. The algebraic-QSS and $K=1,2,3$ dynamic closure
terms are then evaluated at that ensemble-mean state, using
$R_S=R\overline S/S_0$.

1. `python 01_generate_data.py --workers 8`
   Runs the finite-susceptible-pool event-driven ABM (an active-forest
   simulation with exact descendant-depth moment counting, validated
   in-line against an exhaustive frontier-walk cross-check on every step
   of a synthetic run), averages the retained genealogical moments over
   the ensemble at each observation time, and evaluates the zeroth-order
   algebraic QSS closure and the $K=1,2,3$ dynamic tail closures at that
   ensemble-mean state. Pure data generation -- no plotting code, no
   matplotlib import. Writes every numeric result to `data/`. Use
   `--smoke` for a fast end-to-end check with small replicate counts and
   a short time horizon.

2. `python 02_make_figure.py`
   Reads only the cached files in `data/` and draws the two-row,
   four-panel figure (top: closure-supplied vs. ABM-measured local term;
   bottom: signed-defect histograms with three declared ticks each). The
   PDF is written directly to `../../figures/Fig6.pdf`, which is the
   exact path `sn-article.tex` includes
   (`\includegraphics{figures/Fig6.pdf}`) -- re-running this script is
   the only step needed to refresh the manuscript figure. Never
   re-simulates the ABM.

```powershell
pip install -r requirements.txt
python 01_generate_data.py --workers 8
python 02_make_figure.py
```
