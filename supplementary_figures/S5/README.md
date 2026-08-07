# Supplementary Figure S5 - direct event-term verification

The instrumented event-driven ABM directly counts transmission,
spontaneous-removal, identification, and traced-removal events. Over the
same realized paths, it also integrates the ABM-supplied state variables
entering the analytical event intensities beta*U*I, gamma*I, gamma_c*I, and
gamma_c*p_f*M1, and compares observed event counts against these state-
supplied expectations.

Production design: U0=8000, I0=160, 195 crossed raw-rate protocols, 120
independent realizations per protocol, 150 intervals on 0<=tau<=5, tracing
activated at tau=0.5, master seed=20260817.

Run, in order:

    python generate_data_S5.py --threads 5
    python generate_figure_S5.py

`generate_data_S5.py` runs the Gillespie ABM (numba-JIT kernel in
`s5_common.py`) for every protocol and replicate, and writes only numerical
data:
- `supp_S5_event_term_validation_points.csv.gz` -- expected vs. observed event counts for every (protocol, interval) pair (29,250 rows)
- `supp_S5_event_term_validation_protocols.csv` -- the 195 raw-rate protocol definitions
- `supp_S5_event_term_validation_panel_summary.csv` -- per-event calibration statistics (through-origin slope, R^2, RMSE, MAE)
- `supp_S5_event_term_validation_ensemble_data.npz` -- compact per-protocol mean arrays
- `supp_S5_event_term_validation_metadata.json` -- run configuration and the same panel statistics, for provenance

`generate_figure_S5.py` reads those files back, renders the four-panel
scatter figure (PDF/PNG/TIFF at 600 dpi) with `s5_common.plot_panels`, and
writes the caption. It runs no new simulations.

This is a from-scratch, more readable rewrite of the original combined
`generate_supp_S5_event_term_validation.py` (kept in `../event_term_validation/`
for provenance, since it is already referenced by that folder's
`MANIFEST.txt`/`SHA256SUMS.txt` and packaged in `ESM_3.zip`). Both the model
core and the master seed are unchanged, and this folder's regenerated data
has been verified against the archived `event_term_validation/` output with
zero mismatches across all 29,250 points.
