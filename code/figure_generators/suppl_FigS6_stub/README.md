> **Documentation only — no pipeline here.** This figure appears in the
> paper as Supplementary Material 1, Fig. S6 and is produced by
> `supplementary_figures/suppl_FigS6/`.

# Figure 2 -- direct event-level verification of the balance terms

Unlike Figures 1 and 3-9, this figure is not generated from its own
standalone `suppl_FigS6_stub/` pipeline. It is the same asset as Supplementary
Figure S5 (Supplementary Material 1, Sect. S5.6): the instrumented Gillespie ABM
used throughout `suppl_FigS2`-`suppl_FigS5` for the finite-population
benchmark (Sect. 2.7 of the main text) also records raw event counts
(transmission, spontaneous removal, identification, traced removal), and
this figure compares those counts against the corresponding state-supplied
analytical expectations.

Because it shares the 195-protocol production run (`run_full_design.py`,
master seed 20260804) with `suppl_FigS2`-`suppl_FigS5`, duplicating a
generator here would either re-implement that shared dependency or silently
diverge from it. The actual generator, data, and reproduction instructions
live in `../../supplementary_figures/suppl_FigS6/` -- see that folder's `README.md`.

The rendered asset is copied from there into `../../figures/Fig2.pdf` and
`../../figures/Fig2.tiff` (the paths `manuscript.tex` and the BMB TIFF
submission set actually use); re-running `supplementary_figures/suppl_FigS6/`'s
pipeline and re-copying its output is the only step needed to refresh this
figure.

```powershell
cd ../../supplementary_figures/S5
pip install -r requirements.txt
python generate_data_S5.py --threads 5
python generate_figure_S5.py
copy supp_S5_event_term_validation.pdf ..\..\figures\Fig2.pdf
copy supp_S5_event_term_validation.tiff ..\..\figures\Fig2.tiff
```
