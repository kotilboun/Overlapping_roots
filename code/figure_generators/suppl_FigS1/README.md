> **This pipeline produces `Fig4.pdf`, which appears in the paper as Supplementary Material 1, Fig. S1.**
> The output filename keeps the original numbering; the folder name gives
> the destination. See the repository README for the full mapping.

# Figure 4 -- perturbation scale and canonical-QSS approximation accuracy

Two-step, standalone pipeline; no external inputs beyond this folder.

1. `python 01_generate_data.py`
   Solves the canonical selected-QSS continuation (`canonical_qss.py`, an
   adaptive zero-terminal continued fraction that doubles depth until the
   root and Bessel cross-check pass) on a dense c-grid for R in
   {1,1.5,2,4}, computes the closed-form perturbative approximations
   m1^[p] (p=0,1,2) and their errors relative to the canonical branch,
   runs the fixed-depth convergence sidecar (K=20,40,80,160), and sweeps
   R in [0.02,10] at c=1 for panel (e). Pure data generation -- no
   plotting code, no matplotlib import. Writes every numeric result to
   `data/`. Use `--smoke` for coarser grids.

2. `python 02_make_figure.py`
   Reads only the cached files in `data/` (`perturbation_data.npz`,
   `configuration.json`) and draws the five-panel figure. The PDF is
   written directly to `../../figures/Fig4.pdf`, which is the exact path
   `manuscript.tex` includes (`\includegraphics{figures/Fig4.pdf}`) --
   re-running this script is the only step needed to refresh the
   manuscript figure. Never re-solves the canonical QSS branch.

```powershell
pip install -r requirements.txt
python 01_generate_data.py
python 02_make_figure.py
```

`canonical_qss.py` is an unmodified library module imported by
`01_generate_data.py`.
