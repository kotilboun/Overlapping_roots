#!/usr/bin/env python3
"""Step 1/2 for Figure 4: compute the canonical QSS tables and perturbation errors.

Self-contained data generator, with no plotting code. It:

1. solves the canonical selected-QSS continuation (``canonical_qss.py``) on a
   dense c-grid for R in {1, 1.5, 2, 4} -- the adaptive zero-terminal
   continued fraction doubles depth until the root and Bessel cross-check
   pass declared tolerances;
2. computes the closed-form perturbative approximations m1^[p] for p=0,1,2
   and their errors relative to the canonical branch;
3. runs the fixed-depth convergence sidecar (K=20,40,80,160); and
4. sweeps R in [0.02, 10] at c=1 for panel (e).

Every numeric result is written to ``data/``. ``02_make_figure.py`` reads only
those cached files -- it never re-solves anything.

Run:
    python 01_generate_data.py
    python 01_generate_data.py --smoke     # coarser grids, still writes to data/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from canonical_qss import SolverConfig, solve_continuation, solve_fixed_depth_path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

ANALYTICAL_TABLE_ID = "FIG3_CANONICAL_QSS_V2"
CURVE_R_VALUES = (1.0, 1.5, 2.0, 4.0)
CONVERGENCE_C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
CONVERGENCE_DEPTHS = (20, 40, 80, 160)
PANEL_A_R_MAX = 10.0
PANEL_E_R_MIN = 0.02
PANEL_E_R_MAX = 10.0

R_COLORS = {1.0: "#C44E52", 1.5: "#6C55A3", 2.0: "#2878B5", 4.0: "#3C8D55"}
PANEL_E_COLORS = {"canonical_qss": "#1A1A1A", "p0": "#808080", "p1": "#008C95"}


def epsilon(R: float | np.ndarray, c: float | np.ndarray) -> float | np.ndarray:
    return c * R / (R + 1.0) ** 2


def no_tracing_m1(R: float | np.ndarray) -> float | np.ndarray:
    return R / (R + 1.0)


def closed_form_coefficients(R: float) -> np.ndarray:
    return np.asarray(
        [
            1.0 / (R + 2.0),
            (7.0 + 2.0 * R - R**2) / ((R + 2.0) ** 2 * (R + 3.0)),
            (R**5 - 7.0 * R**4 - 52.0 * R**3 - 25.0 * R**2 + 213.0 * R + 254.0)
            / ((R + 2.0) ** 3 * (R + 3.0) ** 2 * (R + 4.0)),
        ],
        dtype=float,
    )


def perturbative_m1(R: float, c: float | np.ndarray, order: int) -> float | np.ndarray:
    if order not in (0, 1, 2, 3):
        raise ValueError("Implemented perturbation orders are 0, 1, 2, and 3.")
    eps = np.asarray(epsilon(R, c), dtype=float)
    value = np.ones_like(eps)
    coefficients = closed_form_coefficients(R)
    for n in range(1, order + 1):
        value += coefficients[n - 1] * eps**n
    output = float(no_tracing_m1(R)) * value
    return float(output) if np.ndim(output) == 0 else output


def leading_omitted_term(R: float, c: float | np.ndarray, retained_order: int) -> float | np.ndarray:
    next_order = retained_order + 1
    coefficients = closed_form_coefficients(R)
    if next_order > len(coefficients):
        output = np.full_like(np.asarray(c, dtype=float), np.nan)
    else:
        output = np.abs(
            float(no_tracing_m1(R)) * coefficients[next_order - 1] * np.asarray(epsilon(R, c), dtype=float) ** next_order
        )
    return float(output) if np.ndim(output) == 0 else output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_configuration(args: argparse.Namespace, solver_config: SolverConfig) -> tuple[dict[str, Any], str]:
    source_files = (
        Path(__file__).resolve(),
        HERE / "canonical_qss.py",
    )
    core: dict[str, Any] = {
        "schema_version": "1.0",
        "analytical_table_id": ANALYTICAL_TABLE_ID,
        "curve_R_values": CURVE_R_VALUES,
        "c_points": args.c_points,
        "c_domain": [0.0, 1.0],
        "panel_A_domain": {"R": ["0+", PANEL_A_R_MAX], "c": [0.0, 1.0]},
        "panel_E_domain": {"R": [PANEL_E_R_MIN, PANEL_E_R_MAX], "c": 1.0},
        "panel_E_R_points": args.panel_e_r_points,
        "panel_E_continuation_c_points": args.panel_e_continuation_c_points,
        "displayed_orders": [0, 1, 2],
        "plot_style": {
            "R_colors": {str(R): R_COLORS[R] for R in CURVE_R_VALUES},
            "panel_e_colors": PANEL_E_COLORS,
        },
        "solver": solver_config.to_dict(),
        "convergence_depths": CONVERGENCE_DEPTHS,
        "convergence_c_values": CONVERGENCE_C_VALUES,
        "sources": {path.name: file_sha256(path) for path in source_files},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    serialized = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(serialized).hexdigest()
    config = dict(core)
    config["configuration_sha256"] = checksum
    return config, checksum


def generate_main_tables(
    c_points: int, solver_config: SolverConfig, configuration_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    c_grid = np.linspace(0.0, 1.0, c_points)
    n_R = len(CURVE_R_VALUES)
    qss = np.zeros((n_R, c_points), dtype=float)
    approximations = np.zeros((n_R, 3, c_points), dtype=float)
    signed_error = np.zeros_like(approximations)
    absolute_error = np.zeros_like(approximations)
    depths = np.zeros((n_R, c_points), dtype=int)
    residuals = np.zeros((n_R, c_points), dtype=float)
    depth_differences = np.zeros((n_R, c_points), dtype=float)
    bessel_differences = np.zeros((n_R, c_points), dtype=float)
    root_margins = np.zeros((n_R, c_points), dtype=float)
    root_error_estimates = np.zeros((n_R, c_points), dtype=float)

    canonical_rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    error_summary_rows: list[dict[str, Any]] = []

    for R_index, R in enumerate(CURVE_R_VALUES):
        results = solve_continuation(R, c_grid, solver_config)
        for c_index, (c, result) in enumerate(zip(c_grid, results, strict=True)):
            canonical_rows.append(
                {"analytical_table_id": ANALYTICAL_TABLE_ID, "configuration_sha256": configuration_sha256, **result.to_row()}
            )
            qss[R_index, c_index] = result.m1
            depths[R_index, c_index] = result.accepted_depth
            residuals[R_index, c_index] = result.fixed_point_residual
            depth_differences[R_index, c_index] = result.depth_doubling_difference
            bessel_differences[R_index, c_index] = result.bessel_cf_difference
            root_margins[R_index, c_index] = result.simple_root_margin
            root_error_estimates[R_index, c_index] = result.a_posteriori_root_error_estimate
            for order in (0, 1, 2):
                approximation = float(perturbative_m1(R, float(c), order))
                signed = approximation - result.m1
                approximations[R_index, order, c_index] = approximation
                signed_error[R_index, order, c_index] = signed
                absolute_error[R_index, order, c_index] = abs(signed)
                perturbation_rows.append(
                    {
                        "analytical_table_id": ANALYTICAL_TABLE_ID,
                        "configuration_sha256": configuration_sha256,
                        "R": R,
                        "c": float(c),
                        "epsilon": float(epsilon(R, float(c))),
                        "retained_order_p": order,
                        "m1_canonical_selected_qss": result.m1,
                        "m1_perturbative": approximation,
                        "signed_error_approximation_minus_qss": signed,
                        "absolute_error_e_p": abs(signed),
                        "first_omitted_term_indicator": float(leading_omitted_term(R, float(c), order)),
                        "canonical_accepted_depth": result.accepted_depth,
                        "canonical_fixed_point_residual": result.fixed_point_residual,
                        "canonical_depth_doubling_difference": result.depth_doubling_difference,
                        "canonical_bessel_cf_difference": result.bessel_cf_difference,
                        "canonical_root_error_estimate": result.a_posteriori_root_error_estimate,
                    }
                )

        for order in (0, 1, 2):
            error = absolute_error[R_index, order]
            location = int(np.argmax(error))
            signed = signed_error[R_index, order]
            changes = np.where(
                (signed[:-1] != 0.0) & (signed[1:] != 0.0) & (np.signbit(signed[:-1]) != np.signbit(signed[1:]))
            )[0]
            crossings = ";".join(f"[{c_grid[index]:.9g},{c_grid[index + 1]:.9g}]" for index in changes if c_grid[index] > 0.0)
            error_summary_rows.append(
                {
                    "scope": "per_R",
                    "R": R,
                    "retained_order_p": order,
                    "maximum_absolute_error": float(error[location]),
                    "maximum_error_c": float(c_grid[location]),
                    "signed_error_zero_crossing_brackets_excluding_c0": crossings,
                }
            )

    for order in (0, 1, 2):
        surface = absolute_error[:, order, :]
        flat_index = int(np.argmax(surface))
        R_index, c_index = np.unravel_index(flat_index, surface.shape)
        error_summary_rows.append(
            {
                "scope": "global_displayed_domain",
                "R": CURVE_R_VALUES[R_index],
                "retained_order_p": order,
                "maximum_absolute_error": float(surface[R_index, c_index]),
                "maximum_error_c": float(c_grid[c_index]),
                "signed_error_zero_crossing_brackets_excluding_c0": "see per_R rows",
            }
        )

    arrays = {
        "R_values": np.asarray(CURVE_R_VALUES),
        "c_grid": c_grid,
        "epsilon": np.asarray([epsilon(R, c_grid) for R in CURVE_R_VALUES]),
        "m1_canonical_selected_qss": qss,
        "m1_perturbative": approximations,
        "signed_error": signed_error,
        "absolute_error": absolute_error,
        "accepted_depth": depths,
        "fixed_point_residual": residuals,
        "depth_doubling_difference": depth_differences,
        "bessel_cf_difference": bessel_differences,
        "simple_root_margin": root_margins,
        "a_posteriori_root_error_estimate": root_error_estimates,
    }
    return canonical_rows, perturbation_rows, error_summary_rows, arrays


def generate_convergence_table(configuration_sha256: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values_by_cell: dict[tuple[float, float], dict[int, float]] = {}
    for R in CURVE_R_VALUES:
        for depth in CONVERGENCE_DEPTHS:
            path = solve_fixed_depth_path(R, CONVERGENCE_C_VALUES, depth, maximum_moment_depth=10)
            for result in path:
                values_by_cell.setdefault((R, result.c), {})[depth] = result.m1
                rows.append(
                    {
                        "analytical_table_id": ANALYTICAL_TABLE_ID,
                        "configuration_sha256": configuration_sha256,
                        "R": R,
                        "c": result.c,
                        "zero_terminal_depth_K": depth,
                        "m1_K": result.m1,
                        "difference_from_previous_depth": np.nan,
                        "fixed_point_residual": result.fixed_point_residual,
                        "stable_bessel_cf_difference": result.bessel_cf_difference,
                        "simple_root_margin": result.simple_root_margin,
                        "admissible": result.admissible,
                        "root_bracket_left": result.bracket_left,
                        "root_bracket_right": result.bracket_right,
                        "bracket_expansions": result.bracket_expansions,
                        "bracket_source": result.bracket_source,
                    }
                )
    for row in rows:
        depth = int(row["zero_terminal_depth_K"])
        previous = depth // 2
        if previous in CONVERGENCE_DEPTHS:
            row["difference_from_previous_depth"] = abs(float(row["m1_K"]) - values_by_cell[(float(row["R"]), float(row["c"]))][previous])
    return rows


def generate_panel_e(
    R_points: int, continuation_c_points: int, solver_config: SolverConfig, configuration_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    R_grid = np.linspace(PANEL_E_R_MIN, PANEL_E_R_MAX, R_points)
    continuation_c = np.linspace(0.0, 1.0, continuation_c_points)
    canonical = np.zeros_like(R_grid)
    depth = np.zeros_like(R_grid, dtype=int)
    residual = np.zeros_like(R_grid)
    bessel_difference = np.zeros_like(R_grid)
    rows: list[dict[str, Any]] = []
    for index, R in enumerate(R_grid):
        result = solve_continuation(float(R), continuation_c, solver_config)[-1]
        canonical[index] = result.m1
        depth[index] = result.accepted_depth
        residual[index] = result.fixed_point_residual
        bessel_difference[index] = result.bessel_cf_difference
        rows.append(
            {
                "analytical_table_id": ANALYTICAL_TABLE_ID,
                "configuration_sha256": configuration_sha256,
                "R": float(R),
                "R_zero_handling": "R>0; left boundary is R_to_0_plus limit",
                "c": 1.0,
                "m1_canonical_selected_qss": result.m1,
                "m1_zeroth_order": float(perturbative_m1(float(R), 1.0, 0)),
                "m1_first_order": float(perturbative_m1(float(R), 1.0, 1)),
                "accepted_depth": result.accepted_depth,
                "fixed_point_residual": result.fixed_point_residual,
                "bessel_cf_difference": result.bessel_cf_difference,
            }
        )
    arrays = {
        "m1_vs_R_grid": R_grid,
        "m1_vs_R_canonical": canonical,
        "m1_vs_R_zeroth_order": np.asarray([perturbative_m1(float(R), 1.0, 0) for R in R_grid]),
        "m1_vs_R_first_order": np.asarray([perturbative_m1(float(R), 1.0, 1) for R in R_grid]),
        "m1_vs_R_accepted_depth": depth,
        "m1_vs_R_fixed_point_residual": residual,
        "m1_vs_R_bessel_cf_difference": bessel_difference,
    }
    return rows, arrays


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-points", type=int, default=1001)
    parser.add_argument("--panel-e-r-points", type=int, default=401)
    parser.add_argument("--panel-e-continuation-c-points", type=int, default=21)
    parser.add_argument("--initial-depth", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=640)
    parser.add_argument("--smoke", action="store_true", help="coarser grids for a fast check")
    args = parser.parse_args()
    if args.smoke:
        args.c_points = 101
        args.panel_e_r_points = 61
        args.panel_e_continuation_c_points = 11
    if args.c_points < 11 or args.panel_e_r_points < 11:
        parser.error("Analytical grids are too small.")
    if args.panel_e_continuation_c_points < 3:
        parser.error("Panel-e continuation needs at least three c values.")
    return args


def main() -> None:
    args = parse_arguments()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    solver_config = SolverConfig(initial_depth=args.initial_depth, max_depth=args.max_depth)
    solver_config.validate()

    config, configuration_sha256 = make_configuration(args, solver_config)
    (DATA_DIR / "configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    canonical_rows, perturbation_rows, error_summary_rows, arrays = generate_main_tables(
        args.c_points, solver_config, configuration_sha256
    )
    convergence_rows = generate_convergence_table(configuration_sha256)
    panel_e_rows, panel_e_arrays = generate_panel_e(
        args.panel_e_r_points, args.panel_e_continuation_c_points, solver_config, configuration_sha256
    )
    arrays.update(panel_e_arrays)

    write_csv(DATA_DIR / "canonical_qss_table.csv", canonical_rows)
    write_csv(DATA_DIR / "perturbation_data.csv", perturbation_rows)
    write_csv(DATA_DIR / "perturbation_error_summary.csv", error_summary_rows)
    write_csv(DATA_DIR / "qss_convergence.csv", convergence_rows)
    write_csv(DATA_DIR / "m1_vs_R_data.csv", panel_e_rows)
    np.savez_compressed(DATA_DIR / "perturbation_data.npz", **arrays)

    print(
        f"Generated canonical Figure 4 data: {len(canonical_rows)} QSS rows, "
        f"{len(perturbation_rows)} perturbation rows, "
        f"configuration={configuration_sha256}."
    )


if __name__ == "__main__":
    main()
