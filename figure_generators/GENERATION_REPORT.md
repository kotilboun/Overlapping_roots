# Figure generation and integration report

Generated: 2026-08-03. Updated: 2026-08-04 (added Figures 7 and 8, Sect. 7.7; later the
same day, renamed the susceptible-population symbol from U to S throughout the
manuscript and every figure -- Figs. 5-8 were re-rendered with the updated axis
labels/captions and their fingerprints below were refreshed accordingly; Figs.
1-4 were untouched and keep their original hashes). Updated again 2026-08-05:
several rounds of text-only manuscript revisions (no figures regenerated, no
hashes affected) shifted the PDF from 46 to 47 pages; Figs. 1-2 are unaffected
and Figs. 3-8 each moved back by one page. Updated again 2026-08-06: the
event-level ABM verification scatter plot formerly shown only as Fig. OR1.5
of Online Resource 1 was promoted to a full main-text figure and inserted at
its correct reading position in Sect. 2.8 (immediately after
Eq.~2.50/Eq.~2.8.1). Because LaTeX numbers figures by source order, giving it
manuscript number "2" required renumbering every subsequent figure by +1
(old Figs. 2-8 -> new Figs. 3-9); the physical asset files, `figure_generators/
Figure_N` folders, all `\label`/`\ref`/`\includegraphics` targets, and every
cross-referencing mention in this figure_generators tree were cascaded to
match. The new `Fig2.pdf` is a copy of `supplementary_figures/S5/
supp_S5_event_term_validation.pdf` (documentation stub in `Figure_2/`, no
independent pipeline); Figs. 1 and 3-9 are the pre-existing figures under
their new filenames/labels, byte-for-byte unchanged. The table below reflects
the current layout.

All nine manuscript figures were regenerated from the standalone scripts in
`figure_generators/Figure_1` through `figure_generators/Figure_9` (`Figure_2`
excepted -- see above). The numerical data generators were run before the
plotting scripts; the plotting scripts read the resulting cached data and
wrote the publication assets directly to `../figures/`. Figures 8 and 9
additionally share the ABM/closure engine in `figure_generators/
figures8_9_shared/trajectory_core.py` and were checked to reproduce Figure 7's
R=4 ABM replicates bit-for-bit (same engine, same master seed 20260815).

## Production commands

Run each pair from its own `Figure_N` directory:

```text
Figure_1: python 01_generate_data.py
          python 02_make_figure.py

Figure_3: python 01_generate_data.py --workers 8
          python 02_make_figure.py

Figure_4: python 01_generate_data.py
          python 02_make_figure.py

Figure_5: python 01_generate_data.py --workers 8
          python 02_make_figure.py

Figure_6: python 01_generate_data.py --workers 8
          python 02_make_figure.py

Figure_7: python 01_generate_data.py --workers 8
          python 02_make_figure.py

Figure_8: python 01_generate_data.py --workers 8
          python 02_make_figure.py

Figure_9: python 01_generate_data.py --workers 8
          python 02_make_figure.py
```

`Figure_2` has no `01_generate_data.py`/`02_make_figure.py` of its own -- it
is produced by `supplementary_figures/S5`'s pipeline (part of the shared
195-protocol production run) and copied into `../figures/Fig2.pdf`/`.tiff`;
see `Figure_2/README.md`.

The installed production environment used NumPy 2.4.2, SciPy 1.17.1,
Matplotlib 3.10.9, and Pillow 12.2.0.

## Output fingerprints

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `Fig1.pdf` | 39,789 | `5c8401d1fe9bebc633e49af68fe3dad413cf5b1cc38edf3e06a90a19e0782de3` |
| `Fig2.pdf` | 218,937 | `30511c1b9f8c1e0c8177331f8ffc6dea80725166e23b1a36a23dcdd8a6365cb4` |
| `Fig3.pdf` | 76,736 | `b3ddfc600faf5b9ded1f78162fceb570adc8241bd98ab193dd4bb351d0ddf686` |
| `Fig4.pdf` | 72,331 | `02d959fc0d07836ed5f7bf918c4e3ecac68d6be418ad48a4ced858e51eecb257` |
| `Fig5.png` | 309,993 | `a465724883f9a40c7c6f774844435c404bd0e1ccf1e76954547d4bb674e25627` |
| `Fig6.pdf` | 130,429 | `a53c761488e6df0eaedc348290c048450c686bc5640a825e62d91b5a83adf47c` |
| `Fig7.pdf` | 33,623 | `934cb9603f3e9b15c6e8a14babd5d5417037d95bd86816ce2185d5808d245d4f` |
| `Fig8.pdf` | 90,913 | `61658aaa72e32fe6b914d892ed869d0a4f3e56ad589858a1fce5566dc621c7b3` |
| `Fig9.pdf` | 86,383 | `81225d76bb1b21eba47019e55b9e3eae45b37e10e62c7aabe75f791e8dbdd5fa` |

## Numerical and structural checks

- Figure 1: the forest structure and descendant counts passed the generator's
  structural check.  The dotted boundary in panel (b) encloses the embedded
  subtree rooted at `j`.
- Figure 2: event-level verification (produced by `supplementary_figures/S5`,
  not this folder). Aggregate observed/expected event-count ratios across all
  195 raw-rate protocols and 150 observation intervals: transmission
  `1.00002`, spontaneous removal `1.00009`, identification `0.99997`, traced
  removal `1.00010`. See `supplement/ESM_1/ESM_1.tex` Sect. OR1-S6 for full
  definitions and per-protocol results.
- Figure 3: all 20 parameter cells contain 120 valid and complete replicates;
  there were no extinctions or censoring events.  The maximum fixed-point
  residual was `2.554e-15`, and the equilibrium gate passed with maximum fitted
  change `0.007612 < 0.01`.
- Figure 4: the production cache contains 4,004 QSS rows and 12,012 perturbation
  rows.  The configuration hash is
  `72c2f02795c9120fd3fd7cf2ad8b371f0d5ccc32a8efacf6b2d0a8ca09cb384e`.
- Figure 5: 804 analytical rows, 241 critical-curve points, and 9,200 ABM paths
  were generated.  The threshold-sign checks passed and the largest cohort
  relative Monte Carlo standard error was `0.4978%`.
- Figure 6: genealogy bookkeeping passed; 80 closure-summary rows were
  generated, with a minimum post-activation alive count of 119.
- Figure 7: genealogy bookkeeping passed; 80 trajectory-error rows were
  generated.  The largest ODE step-halving difference was `1.776e-15`.
- Figure 8: R=4 ABM replicates for all five c-values matched Figure 7's
  cached `data/source_abm_trajectories.npz` bit-for-bit (`np.array_equal`).
- Figure 9: R=4 ABM replicates for all five c-values matched Figure 7's
  cached `data/source_abm_trajectories.npz` bit-for-bit (`np.array_equal`).

## Manuscript integration

The Springer Nature manuscript was rebuilt after generation.  The resulting
`../sn-article.pdf` has 48 pages and contains exactly one caption for every
numbered figure:

| Figure | Manuscript page |
|---:|---:|
| 1 | 7 |
| 2 | 21 |
| 3 | 26 |
| 4 | 29 |
| 5 | 33 |
| 6 | 38 |
| 7 | 39 |
| 8 | 40 |
| 9 | 41 |

All nine caption pages were visually checked at manuscript scale; no figure is
cropped, missing, or placed outside the printable area. Figure 2 is new as of
2026-08-06 (Sect. 2.8, promoted from Online Resource 1's Fig. OR1.5). Figures
8 and 9 were added in Sect. 7.7 alongside Figure 7's closure-trajectory
validation to show representative individual-cell trajectory overlays rather
than only the aggregated error. Repeat this check after future regeneration
with:

```text
python audit_manuscript_figures.py ../sn-article.pdf --render-dir integration_previews
```

The audit script's caption match was tightened (`Fig. N ` followed by an
uppercase letter) once Figures 8 and 9 began cross-referencing Figure 7 in
their own captions -- a bare substring search over-counted "Fig. 7" on pages
where it appears only as an inline cross-reference.

The final LaTeX pass reports no undefined references, undefined citations,
missing graphics, or LaTeX errors. Two non-figure overfull-hbox warnings
remain in the manuscript source (equation blocks at lines 245 and 888),
together with one PDF-bookmark warning; these do not affect figure
generation.
