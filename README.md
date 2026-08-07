# Overlapping roots

Code and reproducibility data for:

> Kotil, S.E. *Overlapping roots: a transmission-tree framework for
> population-level contact-tracing dynamics.*

This repository contains the standalone figure scripts, cached numerical
data, publication-ready figures for every numbered figure in the
manuscript (Figures 1-9), the supplementary agent-based-model (ABM)
validation figures (S1-S5), and the electronic supplementary material.
The main-text manuscript LaTeX source itself is not included here (it's
submitted directly to the journal); this repository is the code and data
that produced it and everything cited from it.

## Layout

- `figure_generators/` -- main-text Figures 1-9. Each `Figure_N/` folder except
  `Figure_2/` is a self-contained two-step pipeline: `01_generate_data.py`
  (simulation/computation, no plotting) followed by `02_make_figure.py` (reads
  the cached data and draws the figure). `Figure_2/` is a documentation-only
  stub -- that figure is produced by `supplementary_figures/S5/`'s pipeline
  and copied into `figures/Fig2.pdf`/`.tiff`. See `figure_generators/README.md`.
- `figures/` -- the final rendered figure assets that `figure_generators/*/
  02_make_figure.py` write to directly (`Fig1.pdf`-`Fig9.pdf`, `Fig5.png`),
  plus a 600-dpi LZW-compressed `FigN.tiff` for every figure -- the format
  Bulletin of Mathematical Biology (Springer) requires for combination
  artwork submission. **pdfLaTeX cannot embed `.tiff` directly**
  ("Unknown graphics extension: .tiff", confirmed against this manuscript),
  so `sn-article.tex` includes the native `Fig*.pdf`/`Fig5.png` to compile
  `sn-article.pdf`; the `Fig*.tiff` files are the separate artwork files to
  upload to the journal's submission system (matched to the manuscript by
  figure number, per standard Springer practice), not files the LaTeX source
  itself reads. `figure_generators/make_bmb_tiffs.py` regenerates them --
  for Figures 1, 3, 4, 6, 7 by rasterizing the exact compiled `Fig*.pdf` at
  600 dpi (so the TIFF matches what's in the manuscript pixel-for-pixel,
  modulo rasterization), for Figure 5 by converting its already-600-dpi PNG;
  Figures 2, 8, and 9 write (or, for Figure 2, supply) their own native
  600-dpi TIFF directly.
- `supplementary_figures/` -- supplementary Figures S1-S5, validating the
  finite-population ABM against a high-depth deterministic reference and
  against direct event-level counts. S1-S4 share one production run,
  `run_full_design.py` (195 raw-rate protocols, master seed 20260804 -- the
  full crossed design Online Resource 1 documents); each `SN/` folder then
  has its own `generate_data_SN.py` / `generate_figure_SN.py` pair reading
  from that run's output. Each figure already includes its own 600-dpi LZW
  TIFF. S1-S4 appear only in Online Resource 1 (`ESM_1.tex`, as Figs.
  OR1.1-OR1.4); S5's output additionally appears as main-text Figure 2 (see
  `figure_generators/Figure_2/README.md`), while its methods and summary
  statistics remain documented in `ESM_1.tex` Sect. OR1-S6. See
  `supplementary_figures/README.md`.
- `supplement/` -- the electronic supplementary material: `ESM_1/ESM_1.tex`
  (Online Resource 1, built entirely from `supplementary_figures/`),
  `ESM_2.xlsx` (Online Resource 2, the 780-row parameterization table and
  published summary statistics), and `ESM_4/ESM_4.tex` (Online Resource 4,
  analytical/derivational, no generated figures or tables). See
  `supplement/README.md`. Online Resource 3 (a packaged code/data archive
  for the journal's submission system) is intentionally not duplicated
  here -- this repository's own `figure_generators/` and
  `supplementary_figures/` folders, above, are that same content
  uncompressed.

Every figure folder is independently reproducible: install that folder's
`requirements.txt`, run the data generator, then the figure generator. Cached
data and rendered figures are included in this package, so unpacking it
reproduces the published figures without rerunning anything; rerunning the
generators regenerates them from scratch (verified to reproduce the
included numerical data, in most cases bit-for-bit -- see each folder's
README and `figure_generators/GENERATION_REPORT.md`).

## Requirements

Python 3.11+ with NumPy, SciPy, Matplotlib, and Pillow (exact per-folder
version floors are in each `requirements.txt`); Figure S5 additionally
requires `numba`.

## Citation

If you use this code, please cite the manuscript above. See the manuscript's
Data Availability and Code Availability statements for the full list of
electronic supplementary materials (Online Resources 1-4).

## License

No license file has been added yet -- all rights reserved by default until
one is chosen and added here.
