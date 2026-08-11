SUPPLEMENTARY MATERIAL 3 - REPRODUCIBILITY ARCHIVE

Article: Overlapping roots: a transmission-tree framework for
population-level contact-tracing dynamics
Journal: Mathematical Biosciences
Author: Seyfullah Enes Kotil
Affiliations: Bogazici University, Department of Molecular Biology and
Genetics, Istanbul, Turkiye; Bahcesehir University, School of Medicine,
Department of Biophysics, Istanbul, Turkiye
Corresponding email: enesseyfullah.kotil@bogazici.edu.tr

Contents
--------
- figure_generators/: standalone generators for the nine figures of the original submission.
  Each Figure_N/ folder (except suppl_FigS6_stub/, a documentation-only stub --
  see its README.md) is a two-step, independently reproducible pipeline:
  01_generate_data.py (simulation/computation, no plotting) followed by
  02_make_figure.py (reads the cached data and draws the figure). Cached
  numerical data, rendered figures, low-resolution previews, and
  manuscript caption text are included for every figure.
  figure_generators/GENERATION_REPORT.md records the production run
  that generated the current figures (commands, SHA-256 output hashes,
  and numerical/structural checks per figure).
  figure_generators/audit_manuscript_figures.py independently checks a
  compiled manuscript PDF for exactly one caption per figure number.
  figure_generators/make_bmb_tiffs.py produces the 600-dpi LZW-compressed
  TIFF submission copies of every figure.
- supplementary_figures/: generators for Supplementary Material 1's own Figures
  S2-S5 (S1-S4, sharing one 195-raw-rate-protocol production run,
  run_full_design.py, master seed 20260804) and the direct event-level
  verification analysis (S5), whose four-panel scatter plot is Fig. S6
  of Supplementary Material 1 (figure_generators/suppl_FigS6_stub/README.md explains this
  relationship). Each figure includes its own 600-dpi LZW TIFF.
- MANIFEST_SHA256.txt: SHA-256 checksums for every archived file.

Figure numbering
----------------
The figure_generators/Figure_N/ folders are numbered by the nine-figure scheme
of the original submission. The article now carries five main-text figures, and
four of the original nine were moved into the supplements. Folder names and file
names are unchanged so that the archived paths, relative imports and checksums
remain valid; use this mapping to locate each figure.

  generator folder    originally    now appears as
  ----------------    ----------    ------------------------------------------
  maintext_Fig1/           Figure 1      main text, Figure 1
  suppl_FigS6_stub/           Figure 2      Supplementary Material 1, Fig. S6
  maintext_Fig2/           Figure 3      main text, Figure 2
  suppl_FigS1/           Figure 4      Supplementary Material 1, Fig. S1
  maintext_Fig3/           Figure 5      main text, Figure 3
  suppl_FigS7/           Figure 6      Supplementary Material 1, Fig. S7
  maintext_Fig4/           Figure 7      main text, Figure 4
  suppl_FigS8/           Figure 8      Supplementary Material 1, Fig. S8
  maintext_Fig5/           Figure 9      main text, Figure 5

Section numbers quoted inside the per-figure README.md files also refer to the
original section numbering.

Reproduction
------------
Each Figure_N/ or SN/ folder is independently reproducible: install that
folder's requirements.txt, run the data generator, then the figure
generator, from inside that folder so relative paths resolve correctly.
The archived rendered figures and cached numerical data are the
submission versions; rerunning the generators reproduces them from
scratch (verified to reproduce the included numerical data, in most
cases bit-for-bit -- see each folder's README and
figure_generators/GENERATION_REPORT.md).

Generated: 2026-08-06

Scope note
----------
This archive is a snapshot of the complete, verified figure-generation
and validation codebase underlying the manuscript and Supplementary Material 1,
current as of the date below. It supersedes any earlier, partial
snapshot of this archive.
