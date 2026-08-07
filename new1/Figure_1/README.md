# Figure 1 -- root-centered bookkeeping in one physical active transmission forest

Two-step, standalone pipeline. Each step is a plain script with no hidden state:

1. `python 01_generate_data.py`
   Reads `forest_spec.json` (the one physical forest all three panels share),
   verifies it is acyclic and that the declared descendant counts D_i^(k) /
   D_j^(k) are correct, and writes the checked result to `data/forest_check.json`.
   A failed structural or descendant-count assertion stops here.

2. `python 02_make_figure.py`
   Reads `forest_spec.json` + `data/forest_check.json` and draws the figure.
   The PDF is written directly to `../../figures/Fig1.pdf`, which is the exact
   path `sn-article.tex` includes (`\includegraphics{figures/Fig1.pdf}`) --
   re-running this script is the only step needed to refresh the manuscript
   figure. A PNG preview and the caption text are also written locally.

```powershell
pip install -r requirements.txt
python 01_generate_data.py
python 02_make_figure.py
```
