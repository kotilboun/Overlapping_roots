# Overlapping roots: a transmission-tree framework for population-level contact-tracing dynamics

Manuscript source, supplementary material, data, and the complete
figure-generation and validation code for the paper.

**Author:** Seyfullah Enes Kotil
([ORCID 0000-0002-9588-3947](https://orcid.org/0000-0002-9588-3947))
Department of Molecular Biology and Genetics, Boğaziçi University, Istanbul, Türkiye ·
School of Medicine, Department of Biophysics, Bahçeşehir University, Istanbul, Türkiye
· enesseyfullah.kotil@bogazici.edu.tr

**Status:** submitted to *Mathematical Biosciences* (Elsevier).

---

## What the paper does

Contact tracing cannot be computed from compartment counts alone: its effect
depends on where detected cases sit in the realized transmission graph. This
work treats **every infectious individual as the root of its own local active
genealogy** and aggregates those overlapping views into population variables
`M_k` counting active descendants at depth *k*.

Three results follow:

- The forward-tracing loss is exactly `−γ_c p_f M₁` — one interpretable count.
- Growth obeys `I′/I = R_S − 1 − c·m₁`: transmission, removal, tracing.
- Maximal one-step tracing reverses early growth only for `R < R_*(1) ≈ 1.67`;
  above that it still cuts future infectious activity by about 40 %.

---

## Repository layout

| folder | contents |
|---|---|
| `manuscript/` | LaTeX source, figures, bibliography, and the built `manuscript.pdf` |
| `supplement/` | Supplementary Material 1: source, figures, and built PDF |
| `data/` | Supplementary Material 2 — parameterizations, error and convergence summaries |
| `code/` | Supplementary Material 3 — figure generators, validation code, cached data |
| `figures_tiff_600dpi/` | 600 dpi TIFF artwork for the five main-text figures |

### Building

The manuscript uses Elsevier's `elsarticle` class; both class and `.bst` are
included, so no journal template download is needed.

```bash
cd manuscript
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript

cd ../supplement
pdflatex supplementary_material_1   # x3
```

The two documents are independent — no build order to observe.

### Reproducing the figures

Each generator folder is a self-contained two-step pipeline:
`01_generate_data.py` (simulation, no plotting) then `02_make_figure.py`
(reads cached data, draws the figure). Install that folder's
`requirements.txt` and run from inside the folder so relative paths resolve.
Cached data and rendered figures are included, so the published figures
reproduce without rerunning any simulation.

Folders are named for where the figure appears — `maintext_Fig1/` … `maintext_Fig5/`
and `suppl_FigS1/` … `suppl_FigS8/`. See the mapping table below.

Python 3.11+ with NumPy, SciPy, Matplotlib and Pillow.
`code/MANIFEST_SHA256.txt` lists SHA-256 checksums for every archived file.

---

## Figure generators

Generator folders are named for **where the figure appears in the paper**.

Main text:

| folder | writes | appears as |
|---|---|---|
| `code/figure_generators/maintext_Fig1/` | `Fig1.pdf` | Figure 1 — root-centered bookkeeping |
| `code/figure_generators/maintext_Fig2/` | `Fig3.pdf` | Figure 2 — constant-pool ABM vs Bessel QSS |
| `code/figure_generators/maintext_Fig3/` | `Fig5.png` | Figure 3 — growth, replacement, control boundary |
| `code/figure_generators/maintext_Fig4/` | `Fig7.pdf` | Figure 4 — closure-trajectory validation |
| `code/figure_generators/maintext_Fig5/` | `Fig9.pdf` | Figure 5 — finite-pool trajectories, algebraic closure |

Supplement:

| folder | writes | appears as |
|---|---|---|
| `code/figure_generators/suppl_FigS1/` | `Fig4.pdf` | Fig. S1 — perturbation scale and accuracy |
| `code/supplementary_figures/suppl_FigS2/` | `supp_S1_convergence_U8000.pdf` | Fig. S2 — ensemble-mean convergence |
| `code/supplementary_figures/suppl_FigS3/` | `supp_S2_selected_production_validation_U8000.pdf` | Fig. S3 — ABM vs hierarchy |
| `code/supplementary_figures/suppl_FigS4/` | `supp_S3_matched_c_raw_rate_comparison_U8000.pdf` | Fig. S4 — matched decompositions |
| `code/supplementary_figures/suppl_FigS5/` | `supp_S4_rate_scale_collapse_U8000.pdf` | Fig. S5 — rate-scale collapse |
| `code/supplementary_figures/suppl_FigS6/` | `supp_S5_event_term_validation.pdf` | Fig. S6 — event-level verification |
| `code/figure_generators/suppl_FigS7/` | `Fig6.pdf` | Fig. S7 — snapshotwise closure terms |
| `code/figure_generators/suppl_FigS8/` | `Fig8.pdf` | Fig. S8 — depth-K trajectory overlays |

Two folders are not pipelines:

- `figure_generators/suppl_FigS6_stub/` — documentation only; Fig. S6 is produced
  by `supplementary_figures/suppl_FigS6/`.
- `figure_generators/shared_trajectory_engine/` — the finite-pool solver shared
  by `maintext_Fig5/` and `suppl_FigS8/`; it must stay their sibling.

**Output filenames keep the original numbering** (`maintext_Fig2/` writes
`Fig3.pdf`). Renaming them would mean rewriting the cached-data references
inside each pipeline, which would invalidate the archived checksums, so the
folder name carries the mapping instead and each folder's `README.md` states it
at the top.

`maintext_Fig5/` and `suppl_FigS8/` read cached data from `maintext_Fig4/`, and
`supplementary_figures/suppl_FigS2`–`suppl_FigS6` read the shared 195-protocol
production run from `supplementary_figures/full_design_data/`. These
dependencies cross the main-text/supplement boundary, which is why the code is
one tree rather than two.

Prose inside the archive's own documentation (`code/README.txt`,
`code/figure_generators/GENERATION_REPORT.md`, and the per-folder
`README.md` files) still quotes the **original** figure and section
numbering where it records the production run. Folder names, file paths
and the tables above use the current numbering.

## Supplementary Material 1 at a glance

| section | contents |
|---|---|
| S1 | Model construction details (supports §2) |
| S2 | Analytical and numerical details for the QSS solution (§3) |
| S3 | Perturbative construction of the QSS approximation (§3) |
| S4 | Technical results and large-population ABM ensemble protocol (§4) |
| S5 | Finite-population agent-based benchmark |
| S6 | Closure validation (§5) |
| S7 | Reproducibility files |

Figures S1–S8 and Tables S1–S10 number continuously through the document.

---

## Citing

Until the paper appears, please cite it as a submitted manuscript:

> S. E. Kotil, *Overlapping roots: a transmission-tree framework for
> population-level contact-tracing dynamics*, submitted to Mathematical
> Biosciences, 2026.

---

## Notes

- **No licence file is included.** Without one, default copyright applies and
  others cannot legally reuse the code. If you want the code reusable, add a
  `LICENSE` — MIT or BSD-3-Clause are common for research code, CC-BY for text
  and figures.
- Generative AI tools (ChatGPT, Claude Code) assisted with document
  organization, language editing, code generation, figure-generation workflows
  and proofreading; all outputs were reviewed, tested and revised by the
  author, who takes full responsibility for the content. This is stated in the
  manuscript's declaration section.
