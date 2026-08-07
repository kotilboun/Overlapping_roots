# Figure 2 -- direct event-level verification of the balance terms

Unlike Figures 1 and 3-9, this figure is not generated from its own
standalone `Figure_2/` pipeline. It is the same asset as Supplementary
Figure S5 (Online Resource 1, Sect. OR1-S6): the instrumented Gillespie ABM
used throughout `supplementary_figures/S1`-`S4` for the finite-population
benchmark (Sect. 2.7 of the main text) also records raw event counts
(transmission, spontaneous removal, identification, traced removal), and
this figure compares those counts against the corresponding state-supplied
analytical expectations.

Because it shares the 195-protocol production run (`run_full_design.py`,
master seed 20260804) with `supplementary_figures/S1`-`S4`, duplicating a
generator here would either re-implement that shared dependency or silently
diverge from it. The actual generator, data, and reproduction instructions
live in `../../supplementary_figures/S5/` -- see that folder's `README.md`.

The rendered asset is copied from there into `../../figures/Fig2.pdf` and
`../../figures/Fig2.tiff` (the paths `sn-article.tex` and the BMB TIFF
submission set actually use); re-running `supplementary_figures/S5/`'s
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
