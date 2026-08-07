#!/usr/bin/env python3
"""Step 1/2 for Figure 3: run the constant-pool ABM simulations and cache the results.

Self-contained data generator, with no plotting code. It:

1. solves the implicit modified-Bessel QSS fixed point at every (R, c) cell and
   cross-checks it against independent K=40/K=80 continued-fraction evaluations;
2. simulates the exact event-driven constant-pool active transmission forest
   (R in {1, 1.5, 2, 4}, c in {0, 0.25, 0.5, 0.75, 1}, 120 realizations per cell);
3. bootstraps replicate-level confidence intervals and checks that the late
   averaging window (3 <= tau <= 3.25) is at equilibrium; and
4. writes every numeric result needed to draw the figure into ``data/``.

``02_make_figure.py`` reads only the files written here -- it never re-simulates.

Run:
    python 01_generate_data.py --workers 8
    python 01_generate_data.py --smoke        # fast check: 4 replicates, writes to data/smoke/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ive

HERE = Path(__file__).resolve().parent

# --- Fixed protocol (matches Sect. 4 / Supplementary Sect. S4.3 of the manuscript) ---
R_VALUES = (1.0, 1.5, 2.0, 4.0)
C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
ORDERS = (1, 2, 3)
I0 = 160
REPLICATES = 120
TAU_END = 3.25
DT = 0.01
PRIMARY_WINDOW = (3.0, 3.25)
SENSITIVITY_WINDOWS = ((2.75, 3.0), (3.0, 3.25))
EQUILIBRIUM_MAX_ABSOLUTE_CHANGE = 0.01
REPRESENTATIVE_CELL = (2.0, 1.0)
BOOTSTRAPS = 5000
SIMULATION_SEED = 2026073101
BOOTSTRAP_SEED = 2026073102
MAXIMUM_ACTIVE = 5_000_000
ROOT_GRID_POINTS = 5001


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(master: int, *coordinates: int) -> int:
    sequence = np.random.SeedSequence([master, *coordinates])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


# --- Selected QSS branch: implicit modified-Bessel fixed point (Sect. 4, Eq. 4.8) ---


def scaled_bessel_ratio(order: float | np.ndarray, z: float) -> np.ndarray:
    denominator = ive(order, z)
    numerator = ive(np.asarray(order) + 1.0, z)
    ratio = np.asarray(numerator / denominator, dtype=float)
    if not np.all(np.isfinite(ratio)):
        raise FloatingPointError("Nonfinite scaled modified-Bessel ratio")
    return ratio


def bessel_fixed_point(R: float, c: float, u: float | np.ndarray) -> np.ndarray:
    u_array = np.asarray(u, dtype=float)
    if c == 0.0:
        return u_array - R / (R + 1.0)
    z = 2.0 * math.sqrt(c * R)
    a = R - c * u_array
    return u_array - math.sqrt(R / c) * scaled_bessel_ratio(a, z)


def bessel_moments(R: float, c: float, u: float) -> np.ndarray:
    if c == 0.0:
        ratios = np.asarray([R / (R + k) for k in ORDERS])
        return np.cumprod(ratios)
    z = 2.0 * math.sqrt(c * R)
    a = R - c * u
    denominator = float(ive(a, z))
    values = [(R / c) ** (k / 2.0) * float(ive(a + k, z)) / denominator for k in ORDERS]
    return np.asarray(values)


def cf_ratios(R: float, c: float, u: float, depth: int) -> np.ndarray:
    ratios = np.zeros(depth + 2, dtype=float)
    for k in range(depth, 0, -1):
        ratios[k] = R / (R + k - c * u + c * ratios[k + 1])
    return ratios


def cf_fixed_point(R: float, c: float, u: float, depth: int) -> float:
    return u - float(cf_ratios(R, c, u, depth)[1])


def roots_on_unit_interval(function) -> list[float]:
    grid = np.linspace(1.0e-10, 1.0 - 1.0e-10, ROOT_GRID_POINTS)
    values = np.asarray(function(grid), dtype=float)
    roots: list[float] = []
    for index in np.where(
        np.isfinite(values[:-1])
        & np.isfinite(values[1:])
        & (np.signbit(values[:-1]) != np.signbit(values[1:]))
    )[0]:
        root = float(
            brentq(
                lambda value: float(function(value)),
                float(grid[index]),
                float(grid[index + 1]),
                xtol=2.0e-14,
                rtol=4.0 * np.finfo(float).eps,
            )
        )
        if not roots or abs(root - roots[-1]) > 1.0e-10:
            roots.append(root)
    near = np.where(np.isfinite(values) & (np.abs(values) < 1.0e-11))[0]
    for index in near:
        root = float(grid[index])
        if all(abs(root - existing) > 5.0e-5 for existing in roots):
            roots.append(root)
    return sorted(roots)


def scalar_cf_roots(R: float, c: float, depth: int) -> list[float]:
    def vectorized(u):
        values = np.asarray(u)
        if values.ndim == 0:
            return cf_fixed_point(R, c, float(values), depth)
        return np.asarray([cf_fixed_point(R, c, float(value), depth) for value in values])

    return roots_on_unit_interval(vectorized)


def calculate_qss_targets() -> tuple[list[dict], dict[tuple[float, float], np.ndarray]]:
    """Solve Eq. 4.8 at every (R, c) and cross-check with K=40/K=80 continued fractions."""
    rows: list[dict] = []
    targets: dict[tuple[float, float], np.ndarray] = {}
    for R in R_VALUES:
        preceding = R / (R + 1.0)
        for c in C_VALUES:
            if c == 0.0:
                roots = [preceding]
                selected = preceding
                moments = bessel_moments(R, c, selected)
                cf40_moments = moments.copy()
                cf80_moments = moments.copy()
                residual = 0.0
            else:
                roots = roots_on_unit_interval(lambda u, R=R, c=c: bessel_fixed_point(R, c, u))
                if not roots:
                    raise RuntimeError(f"No Bessel fixed point at R={R}, c={c}")
                selected = min(roots, key=lambda value: abs(value - preceding))
                moments = bessel_moments(R, c, selected)
                residual = abs(float(bessel_fixed_point(R, c, selected)))
                roots40 = scalar_cf_roots(R, c, 40)
                roots80 = scalar_cf_roots(R, c, 80)
                if not roots40 or not roots80:
                    raise RuntimeError(f"CF fixed point missing at R={R}, c={c}")
                u40 = min(roots40, key=lambda value: abs(value - selected))
                u80 = min(roots80, key=lambda value: abs(value - selected))
                cf40_moments = np.cumprod(cf_ratios(R, c, u40, 40)[1:4])
                cf80_moments = np.cumprod(cf_ratios(R, c, u80, 80)[1:4])

            k_difference = float(np.max(np.abs(cf80_moments - cf40_moments)))
            bessel_difference = float(np.max(np.abs(moments - cf80_moments)))
            if residual >= 1.0e-12:
                raise RuntimeError(f"Bessel residual {residual} fails at R={R}, c={c}")
            if k_difference >= 1.0e-9:
                raise RuntimeError(f"K=40/K=80 difference {k_difference} fails at R={R}, c={c}")
            if bessel_difference >= 1.0e-9:
                raise RuntimeError(f"Bessel/CF difference {bessel_difference} fails at R={R}, c={c}")
            if not (1.0 > moments[0] > moments[1] > moments[2] > 0.0):
                raise RuntimeError(f"Nonadmissible QSS moments at R={R}, c={c}")

            targets[(R, c)] = moments
            rows.append(
                {
                    "R": R,
                    "c": c,
                    "selected_m1_root": selected,
                    "m1_qss": moments[0],
                    "m2_qss": moments[1],
                    "m3_qss": moments[2],
                    "fixed_point_residual": residual,
                    "K40_K80_max_difference_m1_m3": k_difference,
                    "bessel_CF80_max_difference_m1_m3": bessel_difference,
                    "number_of_bessel_roots_found": len(roots),
                    "all_bessel_roots": ";".join(f"{root:.17g}" for root in roots),
                    "selection_rule": "closest to preceding c-continuation value",
                }
            )
            preceding = selected
    return rows, targets


# --- Exact event-driven constant-pool active transmission forest ---


class ActiveGenealogy:
    """Active directed forest with incremental M0,...,M3 bookkeeping."""

    def __init__(self, initial_active: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.kmax = 3
        self.parent: list[int] = []
        self.children: list[list[int]] = []
        self.active: list[bool] = []
        self.active_nodes: list[int] = []
        self.active_position: list[int] = []
        self.totals = np.zeros(4, dtype=np.int64)
        for _ in range(initial_active):
            self._append_root()

    def _append_root(self) -> None:
        node = len(self.parent)
        self.parent.append(-1)
        self.children.append([])
        self.active.append(True)
        self.active_position.append(len(self.active_nodes))
        self.active_nodes.append(node)
        self.totals[0] += 1

    def random_active(self) -> int:
        return self.active_nodes[int(self.rng.integers(len(self.active_nodes)))]

    def add_child(self, parent: int) -> None:
        node = len(self.parent)
        self.parent.append(parent)
        self.children.append([])
        self.active.append(True)
        self.active_position.append(len(self.active_nodes))
        self.active_nodes.append(node)
        self.children[parent].append(node)
        self.totals[0] += 1
        ancestor = parent
        for depth in range(1, self.kmax + 1):
            if ancestor < 0 or not self.active[ancestor]:
                break
            self.totals[depth] += 1
            ancestor = self.parent[ancestor]

    def _path_is_active(self, endpoint: int, depth: int) -> int:
        current = endpoint
        for _ in range(depth + 1):
            if current < 0 or not self.active[current]:
                return 0
            current = self.parent[current]
        return 1

    def _affected_endpoints(self, node: int) -> list[int]:
        endpoints = [node]
        frontier = [node]
        for _ in range(self.kmax):
            following: list[int] = []
            for parent in frontier:
                following.extend(child for child in self.children[parent] if self.active[child])
            endpoints.extend(following)
            frontier = following
            if not frontier:
                break
        return endpoints

    def remove(self, node: int) -> bool:
        if not self.active[node]:
            return False
        endpoints = self._affected_endpoints(node)
        before = np.zeros(4, dtype=np.int64)
        for endpoint in endpoints:
            for depth in range(1, 4):
                before[depth] += self._path_is_active(endpoint, depth)

        self.active[node] = False
        position = self.active_position[node]
        last = self.active_nodes.pop()
        if last != node:
            self.active_nodes[position] = last
            self.active_position[last] = position
        self.active_position[node] = -1
        self.totals[0] -= 1

        after = np.zeros(4, dtype=np.int64)
        for endpoint in endpoints:
            for depth in range(1, 4):
                after[depth] += self._path_is_active(endpoint, depth)
        self.totals[1:] += after[1:] - before[1:]
        return True

    def detect_and_trace(self, node: int, c: float) -> int:
        traced = [child for child in self.children[node] if self.active[child] and self.rng.random() < c]
        self.remove(node)
        removed = 0
        for child in traced:
            removed += int(self.remove(child))
        return removed

    def recompute(self) -> np.ndarray:
        result = np.zeros(4, dtype=np.int64)
        result[0] = len(self.active_nodes)
        for endpoint in self.active_nodes:
            for depth in range(1, 4):
                result[depth] += self._path_is_active(endpoint, depth)
        return result


@dataclass(frozen=True)
class SimulationTask:
    R: float
    c: float
    replicate: int
    seed: int
    times: tuple[float, ...]
    maximum_active: int
    audit: bool


@dataclass
class SimulationResult:
    task: SimulationTask
    counts: np.ndarray
    observed: np.ndarray
    infection_events: int
    identification_events: int
    traced_removals: int
    peak_active: int
    extinction_time: float
    censoring_time: float
    elapsed_seconds: float


def simulate(task: SimulationTask) -> SimulationResult:
    started = time.perf_counter()
    rng = np.random.default_rng(task.seed)
    forest = ActiveGenealogy(I0, rng)
    times = np.asarray(task.times)
    counts = np.full((len(times), 4), np.nan)
    observed = np.zeros(len(times), dtype=bool)
    sample = 0
    now = 0.0
    infections = identifications = traced = 0
    peak = I0
    extinction_time = math.nan
    censoring_time = math.nan

    def record_before(limit: float) -> None:
        nonlocal sample
        while sample < len(times) and times[sample] <= limit + 1.0e-14:
            counts[sample] = forest.totals
            observed[sample] = True
            sample += 1

    while sample < len(times):
        active = len(forest.active_nodes)
        if active == 0:
            extinction_time = now
            break
        if active >= task.maximum_active:
            censoring_time = now
            break
        event_time = now + float(rng.exponential(1.0 / ((task.R + 1.0) * active)))
        record_before(event_time)
        if sample >= len(times):
            break
        now = event_time
        if rng.random() < task.R / (task.R + 1.0):
            forest.add_child(forest.random_active())
            infections += 1
        else:
            traced += forest.detect_and_trace(forest.random_active(), task.c)
            identifications += 1
        peak = max(peak, len(forest.active_nodes))
        totals = forest.totals
        if not (totals[0] >= totals[1] >= totals[2] >= totals[3] >= 0):
            raise RuntimeError("Active-genealogy moment invariant failed")

    if task.audit and not np.array_equal(forest.totals, forest.recompute()):
        raise RuntimeError("Incremental genealogy bookkeeping audit failed")
    return SimulationResult(
        task=task,
        counts=counts,
        observed=observed,
        infection_events=infections,
        identification_events=identifications,
        traced_removals=traced,
        peak_active=peak,
        extinction_time=extinction_time,
        censoring_time=censoring_time,
        elapsed_seconds=time.perf_counter() - started,
    )


def run_simulations(
    times: np.ndarray, replicates: int, workers: int
) -> dict[tuple[float, float], list[SimulationResult]]:
    tasks: list[SimulationTask] = []
    for r_index, R in enumerate(R_VALUES):
        for c_index, c in enumerate(C_VALUES):
            for replicate in range(replicates):
                tasks.append(
                    SimulationTask(
                        R=R,
                        c=c,
                        replicate=replicate,
                        seed=stable_seed(SIMULATION_SEED, r_index, c_index, replicate),
                        times=tuple(float(value) for value in times),
                        maximum_active=MAXIMUM_ACTIVE,
                        audit=replicate == 0,
                    )
                )
    if workers == 1:
        results = [simulate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(simulate, tasks, chunksize=1))
    grouped = {(R, c): [] for R in R_VALUES for c in C_VALUES}
    for result in results:
        grouped[(result.task.R, result.task.c)].append(result)
    for cell in grouped:
        grouped[cell].sort(key=lambda result: result.task.replicate)
    return grouped


def ratio_trajectories(results: list[SimulationResult]) -> np.ndarray:
    counts = np.asarray([result.counts for result in results])
    active = counts[:, :, 0]
    moments = counts[:, :, 1:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = moments / active[:, :, None]
    ratios[~np.isfinite(ratios)] = np.nan
    return ratios


def replicate_window_means(trajectories: np.ndarray, times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    mask = (times >= window[0] - 1.0e-12) & (times <= window[1] + 1.0e-12)
    selected = trajectories[:, mask, :]
    valid_counts = np.sum(np.isfinite(selected), axis=1)
    sums = np.nansum(selected, axis=1)
    return np.divide(sums, valid_counts, out=np.full_like(sums, np.nan), where=valid_counts > 0)


def bootstrap_mean_interval(values: np.ndarray, seed: int, bootstraps: int) -> tuple[float, float, float]:
    valid = np.asarray(values[np.isfinite(values)], dtype=float)
    if valid.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, valid.size, size=(bootstraps, valid.size))
    bootstrap_means = np.mean(valid[indices], axis=1)
    lower, upper = np.percentile(bootstrap_means, (2.5, 97.5))
    return float(np.mean(valid)), float(lower), float(upper)


def bootstrap_trajectory_interval(
    trajectories: np.ndarray, seed: int, bootstraps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    replicates, n_times = trajectories.shape
    valid = np.isfinite(trajectories)
    values = np.nan_to_num(trajectories, nan=0.0)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(replicates, np.full(replicates, 1.0 / replicates), size=bootstraps)
    denominator = weights @ valid.astype(float)
    numerator = weights @ values
    bootstrap_means = np.divide(
        numerator, denominator, out=np.full((bootstraps, n_times), np.nan), where=denominator > 0
    )
    point_count = np.sum(valid, axis=0)
    point_mean = np.divide(
        np.sum(values, axis=0), point_count, out=np.full(n_times, np.nan), where=point_count > 0
    )
    lower = np.nanpercentile(bootstrap_means, 2.5, axis=0)
    upper = np.nanpercentile(bootstrap_means, 97.5, axis=0)
    return point_mean, lower, upper, point_count


def summarize(
    grouped: dict[tuple[float, float], list[SimulationResult]],
    targets: dict[tuple[float, float], np.ndarray],
    times: np.ndarray,
    bootstraps: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict[str, np.ndarray]]:
    replicate_rows: list[dict] = []
    panel_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    cell_count_rows: list[dict] = []
    equilibrium_rows: list[dict] = []
    trajectory_payload: dict[str, np.ndarray] = {
        "times": times,
        "R_values": np.asarray(R_VALUES),
        "c_values": np.asarray(C_VALUES),
    }

    for r_index, R in enumerate(R_VALUES):
        for c_index, c in enumerate(C_VALUES):
            results = grouped[(R, c)]
            trajectories = ratio_trajectories(results)
            primary = replicate_window_means(trajectories, times, PRIMARY_WINDOW)
            key = f"R{R:g}_c{c_tag(c)}"
            trajectory_payload[f"{key}_m1_m3"] = trajectories
            trajectory_payload[f"{key}_observed"] = np.asarray(
                [result.observed for result in results], dtype=bool
            )
            target = targets[(R, c)]
            extinctions = sum(math.isfinite(result.extinction_time) for result in results)
            censorings = sum(math.isfinite(result.censoring_time) for result in results)
            primary_mask = (times >= PRIMARY_WINDOW[0] - 1.0e-12) & (times <= PRIMARY_WINDOW[1] + 1.0e-12)
            valid_primary = np.any(np.isfinite(trajectories[:, primary_mask, 0]), axis=1)
            complete_primary = np.all(np.isfinite(trajectories[:, primary_mask, 0]), axis=1)
            extinction_times = [
                result.extinction_time for result in results if math.isfinite(result.extinction_time)
            ]
            cell_count_rows.append(
                {
                    "R": R,
                    "c": c,
                    "total_replicates": len(results),
                    "valid_primary_replicates": int(np.sum(valid_primary)),
                    "complete_primary_replicates": int(np.sum(complete_primary)),
                    "extinct_before_primary_window": sum(
                        value < PRIMARY_WINDOW[0] for value in extinction_times
                    ),
                    "extinct_by_primary_window_end": sum(
                        value <= PRIMARY_WINDOW[1] for value in extinction_times
                    ),
                    "extinct_by_tau_end": extinctions,
                    "safety_censored_by_tau_end": censorings,
                    "earliest_extinction_time": (min(extinction_times) if extinction_times else None),
                }
            )

            window_times = times[primary_mask]
            for order_index, k in enumerate(ORDERS):
                window_values = trajectories[:, primary_mask, order_index]
                valid_counts = np.sum(np.isfinite(window_values), axis=0)
                ensemble_mean = np.divide(
                    np.nansum(window_values, axis=0),
                    valid_counts,
                    out=np.full(window_times.shape, np.nan),
                    where=valid_counts > 0,
                )
                finite = np.isfinite(ensemble_mean)
                slope = (
                    float(np.polyfit(window_times[finite], ensemble_mean[finite], 1)[0])
                    if np.sum(finite) >= 2
                    else math.nan
                )
                absolute_change = abs(slope) * (PRIMARY_WINDOW[1] - PRIMARY_WINDOW[0])
                equilibrium_rows.append(
                    {
                        "R": R,
                        "c": c,
                        "k": k,
                        "window_start": PRIMARY_WINDOW[0],
                        "window_end": PRIMARY_WINDOW[1],
                        "ols_slope_per_tau": slope,
                        "absolute_fitted_change_over_window": absolute_change,
                        "maximum_allowed_absolute_change": EQUILIBRIUM_MAX_ABSOLUTE_CHANGE,
                        "equilibrium_check_pass": bool(
                            np.isfinite(absolute_change) and absolute_change <= EQUILIBRIUM_MAX_ABSOLUTE_CHANGE
                        ),
                        "minimum_pointwise_valid_replicates": int(np.min(valid_counts)),
                    }
                )

            for replicate, result in enumerate(results):
                for order_index, k in enumerate(ORDERS):
                    replicate_rows.append(
                        {
                            "R": R,
                            "c": c,
                            "replicate": replicate,
                            "k": k,
                            "seed": result.task.seed,
                            "window_start": PRIMARY_WINDOW[0],
                            "window_end": PRIMARY_WINDOW[1],
                            "valid_time_points": int(
                                np.sum(
                                    np.isfinite(
                                        trajectories[
                                            replicate,
                                            (times >= PRIMARY_WINDOW[0]) & (times <= PRIMARY_WINDOW[1]),
                                            order_index,
                                        ]
                                    )
                                )
                            ),
                            "replicate_time_average": primary[replicate, order_index],
                            "extinction_time": result.extinction_time,
                            "censoring_time": result.censoring_time,
                            "peak_active": result.peak_active,
                            "infection_events": result.infection_events,
                            "identification_events": result.identification_events,
                            "traced_removals": result.traced_removals,
                        }
                    )

            for order_index, k in enumerate(ORDERS):
                mean, low, high = bootstrap_mean_interval(
                    primary[:, order_index],
                    stable_seed(BOOTSTRAP_SEED, 1, r_index, c_index, order_index),
                    bootstraps,
                )
                discrepancy = mean - target[order_index]
                qss_row = target[order_index]
                panel_rows.append(
                    {
                        "R": R,
                        "c": c,
                        "k": k,
                        "qss_target": qss_row,
                        "valid_replicate_count": int(np.sum(np.isfinite(primary[:, order_index]))),
                        "extinction_count": extinctions,
                        "censoring_count": censorings,
                        "abm_mean": mean,
                        "bootstrap_95_lower": low,
                        "bootstrap_95_upper": high,
                        "signed_discrepancy": discrepancy,
                        "discrepancy_95_lower": low - qss_row,
                        "discrepancy_95_upper": high - qss_row,
                        "absolute_discrepancy": abs(discrepancy),
                        "absolute_relative_discrepancy": abs(discrepancy) / qss_row,
                    }
                )

            for window_index, window in enumerate(SENSITIVITY_WINDOWS):
                window_means = replicate_window_means(trajectories, times, window)
                for order_index, k in enumerate(ORDERS):
                    mean, low, high = bootstrap_mean_interval(
                        window_means[:, order_index],
                        stable_seed(BOOTSTRAP_SEED, 2, window_index, r_index, c_index, order_index),
                        bootstraps,
                    )
                    sensitivity_rows.append(
                        {
                            "R": R,
                            "c": c,
                            "k": k,
                            "window_start": window[0],
                            "window_end": window[1],
                            "valid_replicate_count": int(np.sum(np.isfinite(window_means[:, order_index]))),
                            "abm_mean": mean,
                            "bootstrap_95_lower": low,
                            "bootstrap_95_upper": high,
                            "qss_target": target[order_index],
                            "signed_discrepancy": mean - target[order_index],
                            "absolute_relative_discrepancy": abs(mean - target[order_index]) / target[order_index],
                        }
                    )

    representative = ratio_trajectories(grouped[REPRESENTATIVE_CELL])
    for order_index, k in enumerate(ORDERS):
        mean, low, high, count = bootstrap_trajectory_interval(
            representative[:, :, order_index],
            stable_seed(BOOTSTRAP_SEED, 3, order_index),
            bootstraps,
        )
        trajectory_payload[f"representative_m{k}_mean"] = mean
        trajectory_payload[f"representative_m{k}_lower"] = low
        trajectory_payload[f"representative_m{k}_upper"] = high
        trajectory_payload[f"representative_m{k}_valid_count"] = count
    return replicate_rows, panel_rows, sensitivity_rows, cell_count_rows, equilibrium_rows, trajectory_payload


def c_tag(c: float) -> str:
    return f"{c:g}".replace(".", "p")


def results_for_manuscript(panel_rows: list[dict], qss_rows: list[dict], grouped) -> dict:
    maximums = {}
    for k in ORDERS:
        row = max((row for row in panel_rows if row["k"] == k), key=lambda row: row["absolute_relative_discrepancy"])
        maximums[f"m{k}"] = {
            "maximum_absolute_relative_discrepancy": row["absolute_relative_discrepancy"],
            "percent": 100.0 * row["absolute_relative_discrepancy"],
            "R": row["R"],
            "c": row["c"],
        }
    extinctions = sum(
        math.isfinite(result.extinction_time) for results in grouped.values() for result in results
    )
    censorings = sum(
        math.isfinite(result.censoring_time) for results in grouped.values() for result in results
    )
    earliest_censoring = min(
        (
            result.censoring_time
            for results in grouped.values()
            for result in results
            if math.isfinite(result.censoring_time)
        ),
        default=None,
    )
    return {
        "maximum_absolute_relative_discrepancies": maximums,
        "maximum_fixed_point_residual": max(row["fixed_point_residual"] for row in qss_rows),
        "maximum_K40_K80_difference": max(row["K40_K80_max_difference_m1_m3"] for row in qss_rows),
        "maximum_bessel_CF80_difference": max(row["bessel_CF80_max_difference_m1_m3"] for row in qss_rows),
        "extinction_count": extinctions,
        "censoring_count": censorings,
        "earliest_censoring_time": earliest_censoring,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--smoke", action="store_true", help="4 replicates, 200 bootstraps, writes to data/smoke/")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data_dir = HERE / "data"
    output_dir = data_dir / "smoke" if args.smoke else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    replicates = 4 if args.smoke else REPLICATES
    bootstraps = 200 if args.smoke else BOOTSTRAPS
    started = time.perf_counter()
    times = np.arange(0.0, TAU_END + DT / 2.0, DT)

    qss_rows, targets = calculate_qss_targets()
    grouped = run_simulations(times, replicates, max(1, args.workers))
    (
        replicate_rows,
        panel_rows,
        sensitivity_rows,
        cell_count_rows,
        equilibrium_rows,
        trajectory_payload,
    ) = summarize(grouped, targets, times, bootstraps)
    qss_lookup = {(float(row["R"]), float(row["c"])): row for row in qss_rows}
    for row in panel_rows:
        diagnostic = qss_lookup[(float(row["R"]), float(row["c"]))]
        row.update(
            {
                "root_residual": diagnostic["fixed_point_residual"],
                "K40_K80_max_difference_m1_m3": diagnostic["K40_K80_max_difference_m1_m3"],
                "bessel_CF80_max_difference_m1_m3": diagnostic["bessel_CF80_max_difference_m1_m3"],
                "number_of_roots_found": diagnostic["number_of_bessel_roots_found"],
            }
        )

    write_csv(output_dir / "qss_targets.csv", qss_rows)
    write_csv(output_dir / "replicate_window_summaries.csv", replicate_rows)
    write_csv(output_dir / "panel_summary.csv", panel_rows)
    write_csv(output_dir / "window_sensitivity.csv", sensitivity_rows)
    write_csv(output_dir / "cell_counts.csv", cell_count_rows)
    write_csv(output_dir / "equilibrium_check.csv", equilibrium_rows)

    equilibrium_pass = all(bool(row["equilibrium_check_pass"]) for row in equilibrium_rows)
    if not args.smoke and not equilibrium_pass:
        failed = [
            (row["R"], row["c"], row["k"], row["absolute_fitted_change_over_window"])
            for row in equilibrium_rows
            if not row["equilibrium_check_pass"]
        ]
        raise RuntimeError("Late-window equilibrium criterion failed: " + repr(failed))

    np.savez_compressed(output_dir / "trajectories.npz", **trajectory_payload)

    manuscript_results = results_for_manuscript(panel_rows, qss_rows, grouped)
    manuscript_results["equilibrium_check"] = {
        "window": PRIMARY_WINDOW,
        "maximum_allowed_absolute_fitted_change": EQUILIBRIUM_MAX_ABSOLUTE_CHANGE,
        "maximum_observed_absolute_fitted_change": max(
            row["absolute_fitted_change_over_window"] for row in equilibrium_rows
        ),
        "all_cells_and_coordinates_pass": equilibrium_pass,
    }
    manuscript_results["cell_counts"] = cell_count_rows
    (output_dir / "results_for_manuscript.json").write_text(
        json.dumps(manuscript_results, indent=2) + "\n", encoding="utf-8"
    )

    metadata = {
        "status": "production" if not args.smoke else "smoke",
        "protocol": {
            "R": R_VALUES,
            "c": C_VALUES,
            "I0": I0,
            "replicates_per_cell": replicates,
            "tau_end": TAU_END,
            "dt": DT,
            "primary_window": PRIMARY_WINDOW,
            "representative_cell": REPRESENTATIVE_CELL,
            "equilibrium_max_absolute_fitted_change": EQUILIBRIUM_MAX_ABSOLUTE_CHANGE,
            "bootstrap_resamples": bootstraps,
            "simulation_seed": SIMULATION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "maximum_active_safety_cap": MAXIMUM_ACTIVE,
            "workers": args.workers,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manuscript_results, indent=2))


if __name__ == "__main__":
    main()
