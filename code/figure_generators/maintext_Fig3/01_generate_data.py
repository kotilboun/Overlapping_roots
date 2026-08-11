#!/usr/bin/env python3
"""Step 1/2 for Figure 5: compute the analytical curves and run the ABM validation.

Self-contained data generator, with no plotting code. It:

1. solves the canonical selected-QSS continuation (``canonical_qss.py``) for
   R in {1, 1.5, 2, 4} to get the Malthusian growth rate g(R,c)=a-1 and the
   incident-cohort lifetime reproduction R_time(R,c), and computes the actual
   control boundary c*(R) (where g=0) up to its c=1 endpoint;
2. simulates the constant-pool one-step forward-tracing Markov jump process
   (an event-driven active-forest ABM, independent of the QSS solver) to
   validate those curves:
   - CP_COHORT_R2: complete-follow-up incident-cohort lifetime estimates on
     the same (R,c) grid as the analytical curves;
   - CP_THRESHOLD_R1: event/person-time growth estimates on both sides of the
     analytical control boundary, each replicated at two initial-population
     sizes as a sensitivity check;
3. bootstraps a critical-R estimate (with 95% CI) at each traced c value by
   linearly interpolating the two bracketing cells' growth estimates.

Every numeric result is written to ``data/``. ``02_make_figure.py`` reads only
those cached files -- it never re-solves or re-simulates anything.

The measurement-window start times used to seed each cohort simulation come
from ``figure3_measurement_windows.json`` (Figure 3's per-(R,c)-cell adaptive
equilibration protocol) -- a static, pre-computed input, not regenerated here.

Run:
    python 01_generate_data.py
    python 01_generate_data.py --smoke --workers 4   # small replicate counts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

from canonical_qss import (
    SolverConfig as CanonicalSolverConfig,
    continued_fraction_tail,
    solve_continuation,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
WINDOWS_PATH = HERE / "figure3_measurement_windows.json"

R_CURVES = (1.0, 1.5, 2.0, 4.0)

COHORT_DATASET_ID = "CP_COHORT_R2"
COHORT_C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
COHORT_CELLS = tuple((R, c) for R in R_CURVES for c in COHORT_C_VALUES)
BOUNDARY_C_VALUES = (0.25, 0.5, 0.75, 1.0)
BOUNDARY_OFFSET_R = 0.08
WINDOW_DURATION = 0.10
SAMPLE_STEP = 0.025
COHORT_TARGET_ACTIVE = 4_000
COHORT_MINIMUM_INITIAL = 50
COHORT_MAXIMUM_INITIAL = 20_000
THRESHOLD_INITIAL = 2_000
THRESHOLD_SENSITIVITY_INITIAL = 500
MAXIMUM_ACTIVE = 300_000
BOOTSTRAPS = 4_000
PRODUCTION_SEED = 202607240402
SENSITIVITY_SEED = 202607240403
BOOTSTRAP_SEED = 202607240404
CRITICAL_POINT_BOOTSTRAP_SEED = 202607240405


# --------------------------------------------------------------------------
# Part 1: canonical analytical curves and control boundary (deterministic).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticalConfig:
    base_depth: int = 20
    max_depth: int = 640
    depth_tolerance: float = 2.0e-13
    root_tolerance: float = 2.0e-12
    bessel_tolerance: float = 2.0e-12
    c_points: int = 201
    critical_points: int = 241


def make_shared_solver_config(config: AnalyticalConfig) -> CanonicalSolverConfig:
    shared = CanonicalSolverConfig(
        initial_depth=config.base_depth,
        max_depth=config.max_depth,
        depth_absolute_tolerance=config.depth_tolerance,
        depth_relative_tolerance=0.0,
        fixed_point_residual_tolerance=config.root_tolerance,
        bessel_crosscheck_tolerance=config.bessel_tolerance,
        root_xtol=min(config.root_tolerance, 1.0e-14),
        maximum_reported_moment_depth=18,
    )
    shared.validate()
    return shared


def lifetime_quantities(c: float, moments: np.ndarray) -> tuple[float, float]:
    """Return R_time-related series sum and the B factor (see manuscript Sec. 6)."""

    rtime_sum = 1.0
    B = 1.0
    coefficient = 1.0
    for q in range(1, len(moments)):
        coefficient *= -c / (q + 1.0)
        rtime_sum += coefficient * moments[q - 1]
        B += coefficient * moments[q]
    return rtime_sum, B


def critical_residual_at_depth(R: float, c: float, depth: int) -> float:
    """g=0 residual with a fixed at one."""

    y, _ = continued_fraction_tail(1.0, c * R, depth)
    return R - 1.0 - float(y[1])


def critical_c_at_depth(R: float, depth: int) -> float:
    if abs(R - 1.0) <= 2.0e-14:
        return 0.0
    f0 = critical_residual_at_depth(R, 0.0, depth)
    f1 = critical_residual_at_depth(R, 1.0, depth)
    if f0 < -1.0e-13 or f1 > 1.0e-12:
        raise ValueError(f"No critical intensity in [0,1] for R={R:.15g}")
    if abs(f1) <= 5.0e-14:
        return 1.0
    return brentq(lambda c: critical_residual_at_depth(R, c, depth), 0.0, 1.0, xtol=5.0e-15, rtol=1.0e-14, maxiter=200)


def endpoint_R_at_depth(depth: int) -> float:
    return brentq(lambda R: critical_residual_at_depth(R, 1.0, depth), 1.0, 2.0, xtol=5.0e-15, rtol=1.0e-14, maxiter=200)


def adaptive_endpoint(config: AnalyticalConfig) -> tuple[float, int, float]:
    depth = config.base_depth
    previous = endpoint_R_at_depth(depth)
    while depth < config.max_depth:
        depth = min(2 * depth, config.max_depth)
        current = endpoint_R_at_depth(depth)
        difference = abs(current - previous)
        previous = current
        if difference <= config.depth_tolerance:
            return current, depth, difference
    raise RuntimeError("Critical endpoint did not converge with depth")


def adaptive_critical_c(R: float, config: AnalyticalConfig) -> tuple[float, int, float]:
    depth = config.base_depth
    previous = critical_c_at_depth(R, depth)
    while depth < config.max_depth:
        depth = min(2 * depth, config.max_depth)
        current = critical_c_at_depth(R, depth)
        difference = abs(current - previous)
        previous = current
        if difference <= config.depth_tolerance:
            return current, depth, difference
    raise RuntimeError(f"Critical c did not converge for R={R:g}")


def generate_tables(config: AnalyticalConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    shared_config = make_shared_solver_config(config)
    analytical: list[dict[str, Any]] = []
    for R in R_CURVES:
        continuation = solve_continuation(R, np.linspace(0.0, 1.0, config.c_points), shared_config)
        for result in continuation:
            moments = np.asarray(result.moments, dtype=float)
            rtime_sum, B_factor = lifetime_quantities(result.c, moments)
            R_time = result.R * rtime_sum
            growth_rate = result.a - 1.0
            analytical.append(
                {
                    "R": result.R,
                    "c": result.c,
                    "a": result.a,
                    "m1_qss": result.m1,
                    "growth_rate": growth_rate,
                    "R_time": R_time,
                    "B_factor": B_factor,
                    "identity_residual": abs((R_time - 1.0) - growth_rate * B_factor),
                    "fixed_point_residual": result.fixed_point_residual,
                    "continued_fraction_depth": result.accepted_depth,
                    "bessel_cf_difference": result.bessel_cf_difference,
                    "simple_root_margin": result.simple_root_margin,
                    "admissibility_margin": min(result.minimum_moment, result.minimum_monotonicity_margin),
                    "admissible": result.admissible,
                }
            )

    endpoint_R, endpoint_depth, endpoint_depth_difference = adaptive_endpoint(config)
    critical: list[dict[str, Any]] = []
    for R in np.linspace(1.0, endpoint_R, config.critical_points):
        c_star, depth, depth_difference = adaptive_critical_c(float(R), config)
        check_grid = np.asarray([0.0]) if c_star == 0.0 else np.asarray([0.0, c_star])
        check = solve_continuation(float(R), check_grid, shared_config)[-1]
        rtime_sum, B_factor = lifetime_quantities(check.c, np.asarray(check.moments, dtype=float))
        R_time = check.R * rtime_sum
        growth_rate = check.a - 1.0
        critical.append(
            {
                "R": float(R),
                "c_star": c_star,
                "growth_rate_check": growth_rate,
                "qss_identity_residual": abs((R_time - 1.0) - growth_rate * B_factor),
                "qss_admissibility_margin": min(check.minimum_moment, check.minimum_monotonicity_margin),
                "continued_fraction_depth": depth,
                "depth_c_difference": depth_difference,
            }
        )

    diagnostics = {
        "max_fixed_point_residual": max(row["fixed_point_residual"] for row in analytical),
        "max_bessel_cf_difference": max(row["bessel_cf_difference"] for row in analytical),
        "max_growth_replacement_identity_residual": max(row["identity_residual"] for row in analytical),
        "minimum_admissibility_margin": min(row["admissibility_margin"] for row in analytical),
        "minimum_simple_root_margin": min(row["simple_root_margin"] for row in analytical),
        "all_rows_admissible": all(row["admissible"] for row in analytical),
        "critical_endpoint_R_at_c1": endpoint_R,
        "critical_endpoint_depth": endpoint_depth,
        "critical_endpoint_depth_difference": endpoint_depth_difference,
        "max_critical_growth_residual": max(abs(row["growth_rate_check"]) for row in critical),
        "critical_curve_gridwise_monotone": bool(np.all(np.diff([row["c_star"] for row in critical]) >= -1.0e-12)),
    }
    return analytical, critical, diagnostics


def analytical_lookup(analytical_rows: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, float]]:
    return {(row["R"], row["c"]): {"growth_rate": row["growth_rate"], "R_time": row["R_time"]} for row in analytical_rows}


def critical_R_interpolator(critical_rows: list[dict[str, Any]]):
    c_values = np.asarray([row["c_star"] for row in critical_rows])
    R_values = np.asarray([row["R"] for row in critical_rows])
    return lambda c: float(np.interp(c, c_values, R_values))


# --------------------------------------------------------------------------
# Part 2: constant-pool one-step forward-tracing ABM (stochastic validation).
# --------------------------------------------------------------------------


def stable_seed(master: int, phase: int, R: float, c: float, replicate: int) -> int:
    sequence = np.random.SeedSequence([master, phase, int(round(R * 1_000_000)), int(round(c * 1_000_000)), replicate])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


class ActiveForest:
    """Minimal active forest for the one-step forward-tracing event process."""

    def __init__(self, initial: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.parent: list[int] = []
        self.children: list[list[int]] = []
        self.active: list[bool] = []
        self.active_nodes: list[int] = []
        self.active_position: list[int] = []
        for _ in range(initial):
            self._append(-1)

    def _append(self, parent: int) -> int:
        node = len(self.parent)
        self.parent.append(parent)
        self.children.append([])
        self.active.append(True)
        self.active_position.append(len(self.active_nodes))
        self.active_nodes.append(node)
        if parent >= 0:
            self.children[parent].append(node)
        return node

    def random_active(self) -> int:
        return self.active_nodes[int(self.rng.integers(len(self.active_nodes)))]

    def add_child(self, parent: int) -> int:
        return self._append(parent)

    def remove(self, node: int) -> bool:
        if not self.active[node]:
            return False
        self.active[node] = False
        position = self.active_position[node]
        last = self.active_nodes.pop()
        if last != node:
            self.active_nodes[position] = last
            self.active_position[last] = position
        self.active_position[node] = -1
        return True

    def detect_and_trace(self, node: int, c: float) -> list[int]:
        traced = [child for child in self.children[node] if self.active[child] and self.rng.random() < c]
        removed: list[int] = []
        if self.remove(node):
            removed.append(node)
        for child in traced:
            if self.remove(child):
                removed.append(child)
        return removed

    def active_ancestor_count(self, node: int) -> int:
        count = 0
        current = self.parent[node]
        while current >= 0 and self.active[current]:
            count += 1
            current = self.parent[current]
        return count

    def normalized_moments(self, maximum_depth: int = 3) -> tuple[float, ...]:
        active_count = len(self.active_nodes)
        if active_count == 0:
            return tuple(math.nan for _ in range(maximum_depth))
        totals = [0] * maximum_depth
        for endpoint in self.active_nodes:
            current = endpoint
            for depth in range(maximum_depth):
                current = self.parent[current]
                if current < 0 or not self.active[current]:
                    break
                totals[depth] += 1
        return tuple(total / active_count for total in totals)


@dataclass(frozen=True)
class Task:
    dataset: str
    phase: str
    R: float
    c: float
    side: str
    replicate: int
    seed: int
    initial: int
    entry_start: float
    entry_end: float
    maximum_active: int
    track_cohort: bool


@dataclass
class Result:
    task: Task
    initial_at_window: int
    final_at_window: int
    parent_person_time: float
    infection_events: int
    primary_removal_events: int
    traced_removals: int
    sample_times: tuple[float, ...]
    sample_active: tuple[int, ...]
    m1: float
    m2: float
    m3: float
    cohort_births: int
    cohort_completed: int
    cohort_lifetime_sum: float
    cohort_mean_active_ancestors: float
    cap_time: float
    extinction_time: float


def simulate(task: Task) -> Result:
    rng = np.random.default_rng(task.seed)
    forest = ActiveForest(task.initial, rng)
    now = 0.0
    parent_time = 0.0
    infections = removals = traced = 0
    initial_at_window = final_at_window = -1
    cap_time = extinction_time = math.nan
    sample_times = tuple(
        float(x) for x in np.round(np.arange(task.entry_start, task.entry_end + SAMPLE_STEP / 2.0, SAMPLE_STEP), 12)
    )
    sample_active: list[int] = []
    sample_index = 0
    tracked_birth: dict[int, float] = {}
    tracked_active: set[int] = set()
    completed_lifetimes: list[float] = []
    ancestor_counts: list[int] = []

    def record_until(limit: float) -> None:
        nonlocal sample_index, initial_at_window
        while sample_index < len(sample_times) and sample_times[sample_index] <= limit + 1e-14:
            sample_active.append(len(forest.active_nodes))
            if sample_index == 0:
                initial_at_window = len(forest.active_nodes)
            sample_index += 1

    while now < task.entry_end:
        active_count = len(forest.active_nodes)
        if active_count == 0:
            extinction_time = now
            record_until(task.entry_end)
            break
        if active_count >= task.maximum_active:
            cap_time = now
            break
        next_time = now + float(rng.exponential(1.0 / ((task.R + 1.0) * active_count)))
        record_until(min(next_time, task.entry_end))
        overlap_start = max(now, task.entry_start)
        overlap_end = min(next_time, task.entry_end)
        if overlap_end > overlap_start:
            parent_time += active_count * (overlap_end - overlap_start)
        if next_time > task.entry_end:
            now = task.entry_end
            break
        now = next_time
        in_window = now >= task.entry_start
        if rng.random() < task.R / (task.R + 1.0):
            parent = forest.random_active()
            child = forest.add_child(parent)
            if in_window:
                infections += 1
                if task.track_cohort:
                    tracked_birth[child] = now
                    tracked_active.add(child)
                    ancestor_counts.append(forest.active_ancestor_count(child))
        else:
            removed = forest.detect_and_trace(forest.random_active(), task.c)
            if in_window:
                removals += 1
                traced += max(0, len(removed) - 1)
            if task.track_cohort:
                for node in removed:
                    birth = tracked_birth.get(node)
                    if node in tracked_active and birth is not None:
                        tracked_active.remove(node)
                        completed_lifetimes.append(now - birth)

    if not math.isfinite(cap_time):
        record_until(task.entry_end)
        final_at_window = len(forest.active_nodes)
    if task.track_cohort and not math.isfinite(cap_time):
        # Complete follow-up with births disabled; valid because post-entry
        # births cannot change an enrolled individual's removal/tracing clock.
        while tracked_active:
            active_count = len(forest.active_nodes)
            if active_count == 0:
                raise RuntimeError("Tracked cohort remained after forest extinction")
            now += float(rng.exponential(1.0 / active_count))
            removed = forest.detect_and_trace(forest.random_active(), task.c)
            for node in removed:
                birth = tracked_birth.get(node)
                if node in tracked_active and birth is not None:
                    tracked_active.remove(node)
                    completed_lifetimes.append(now - birth)

    if not math.isfinite(cap_time):
        net_events = infections - removals - traced
        if final_at_window - initial_at_window != net_events:
            raise AssertionError(f"Event-count identity failed: {final_at_window}-{initial_at_window}!={net_events}")
    if task.track_cohort and len(completed_lifetimes) != len(tracked_birth):
        raise AssertionError("Incomplete cohort follow-up")

    moments = forest.normalized_moments(3) if sample_index else (math.nan, math.nan, math.nan)
    return Result(
        task=task,
        initial_at_window=initial_at_window,
        final_at_window=final_at_window,
        parent_person_time=parent_time,
        infection_events=infections,
        primary_removal_events=removals,
        traced_removals=traced,
        sample_times=sample_times,
        sample_active=tuple(sample_active),
        m1=moments[0],
        m2=moments[1],
        m3=moments[2],
        cohort_births=len(tracked_birth),
        cohort_completed=len(completed_lifetimes),
        cohort_lifetime_sum=float(np.sum(completed_lifetimes)),
        cohort_mean_active_ancestors=(float(np.mean(ancestor_counts)) if ancestor_counts else math.nan),
        cap_time=cap_time,
        extinction_time=extinction_time,
    )


def run_tasks(tasks: list[Task], workers: int) -> list[Result]:
    if workers == 1:
        return [simulate(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(simulate, tasks, chunksize=1))


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAPS)
    n = len(numerator)
    for start in range(0, BOOTSTRAPS, 250):
        count = min(250, BOOTSTRAPS - start)
        indices = rng.integers(0, n, size=(count, n))
        estimates[start : start + count] = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def result_row(result: Result) -> dict[str, Any]:
    task = result.task
    net = result.infection_events - result.primary_removal_events - result.traced_removals
    return {
        "dataset": task.dataset,
        "phase": task.phase,
        "R": task.R,
        "c": task.c,
        "boundary_side": task.side,
        "replicate": task.replicate,
        "initial_infectious": task.initial,
        "window_start": task.entry_start,
        "window_end": task.entry_end,
        "parent_person_time": result.parent_person_time,
        "infection_events": result.infection_events,
        "primary_removal_events": result.primary_removal_events,
        "traced_removals": result.traced_removals,
        "net_events": net,
        "cohort_lifetime_sum": result.cohort_lifetime_sum,
        "cap_time": result.cap_time,
        "extinction_time": result.extinction_time,
    }


def summarize(
    results: list[Result], analytical: dict[tuple[float, float], dict[str, float]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, float, float, str, int], list[Result]] = {}
    for result in results:
        task = result.task
        grouped.setdefault((task.dataset, task.phase, task.R, task.c, task.side, task.initial), []).append(result)
    threshold_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for key, cell in sorted(grouped.items()):
        dataset, phase, R, c, side, initial = key
        if any(math.isfinite(item.cap_time) for item in cell):
            raise RuntimeError(f"Cap censoring in {key}")
        person_time = np.asarray([item.parent_person_time for item in cell])
        net = np.asarray(
            [item.infection_events - item.primary_removal_events - item.traced_removals for item in cell], dtype=float
        )
        growth = float(net.sum() / person_time.sum())
        g_low, g_high = bootstrap_ratio(net, person_time, stable_seed(BOOTSTRAP_SEED, 1, R, c, initial))
        threshold_rows.append(
            {
                "dataset": dataset,
                "phase": phase,
                "R": R,
                "c": c,
                "boundary_side": side,
                "initial_infectious": initial,
                "replicates": len(cell),
                "event_growth_estimate": growth,
                "bootstrap_95_low": g_low,
                "bootstrap_95_high": g_high,
                "analytical_growth": analytical[(R, c)]["growth_rate"] if (R, c) in analytical else math.nan,
                "extinctions_before_window_end": sum(
                    math.isfinite(item.extinction_time) and item.extinction_time <= item.task.entry_end for item in cell
                ),
                "total_person_time": float(person_time.sum()),
            }
        )
        if any(item.task.track_cohort for item in cell):
            numerator = np.asarray([item.cohort_lifetime_sum for item in cell])
            pooled = float(numerator.sum() / person_time.sum())
            low, high = bootstrap_ratio(numerator, person_time, stable_seed(BOOTSTRAP_SEED, 2, R, c, initial))
            ratios = numerator / person_time
            target = analytical[(R, c)]["R_time"]
            cohort_rows.append(
                {
                    "dataset": COHORT_DATASET_ID,
                    "phase": phase,
                    "R": R,
                    "c": c,
                    "initial_infectious": initial,
                    "replicates": len(cell),
                    "pooled_R_time": pooled,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "analytical_R_time": target,
                    "difference_from_analytical": pooled - target,
                    "relative_MCSE": float(ratios.std(ddof=1) / math.sqrt(len(ratios)) / max(abs(pooled), 1.0e-15)),
                    "total_parent_person_time": float(person_time.sum()),
                    "total_cohort_births": int(sum(item.cohort_births for item in cell)),
                    "total_completed_cohort": int(sum(item.cohort_completed for item in cell)),
                    "incomplete_cohort": int(sum(item.cohort_births - item.cohort_completed for item in cell)),
                }
            )
    return threshold_rows, cohort_rows


def cohort_initial(R: float, c: float, entry_start: float, growth: float) -> int:
    estimate = round(COHORT_TARGET_ACTIVE * math.exp(-growth * entry_start))
    return int(np.clip(estimate, COHORT_MINIMUM_INITIAL, COHORT_MAXIMUM_INITIAL))


def load_window_starts() -> dict[tuple[float, float], float]:
    protocol = json.loads(WINDOWS_PATH.read_text(encoding="utf-8"))
    return {(float(row["R"]), float(row["c"])): float(row["measurement_window"]["start"]) for row in protocol["cell_windows"]}


def build_tasks(
    analytical: dict[tuple[float, float], dict[str, float]],
    critical_R: Any,
    *,
    cohort_replicates: int,
    threshold_replicates: int,
    sensitivity_replicates: int,
) -> list[Task]:
    starts = load_window_starts()
    tasks: list[Task] = []
    for R, c in COHORT_CELLS:
        entry_start = starts[(R, c)]
        entry_end = entry_start + WINDOW_DURATION
        growth = analytical[(R, c)]["growth_rate"]
        initial = cohort_initial(R, c, entry_start, growth)
        for replicate in range(cohort_replicates):
            tasks.append(
                Task(
                    COHORT_DATASET_ID,
                    "PRODUCTION",
                    R,
                    c,
                    "curve_cell",
                    replicate,
                    stable_seed(PRODUCTION_SEED, 1, R, c, replicate),
                    initial,
                    entry_start,
                    entry_end,
                    MAXIMUM_ACTIVE,
                    True,
                )
            )

    for c in BOUNDARY_C_VALUES:
        center = critical_R(c)
        for side, sign in (("below", -1.0), ("above", 1.0)):
            R = round(center + sign * BOUNDARY_OFFSET_R, 8)
            entry_start = 3.6
            entry_end = entry_start + WINDOW_DURATION
            for replicate in range(threshold_replicates):
                tasks.append(
                    Task(
                        "CP_THRESHOLD_R1",
                        "PRODUCTION",
                        R,
                        c,
                        side,
                        replicate,
                        stable_seed(PRODUCTION_SEED, 2, R, c, replicate),
                        THRESHOLD_INITIAL,
                        entry_start,
                        entry_end,
                        MAXIMUM_ACTIVE,
                        False,
                    )
                )
            for replicate in range(sensitivity_replicates):
                tasks.append(
                    Task(
                        "CP_THRESHOLD_R1",
                        "INITIAL_SIZE_SENSITIVITY",
                        R,
                        c,
                        side,
                        replicate,
                        stable_seed(SENSITIVITY_SEED, 3, R, c, replicate),
                        THRESHOLD_SENSITIVITY_INITIAL,
                        entry_start,
                        entry_end,
                        MAXIMUM_ACTIVE,
                        False,
                    )
                )
    return tasks


# --------------------------------------------------------------------------
# Part 3: bootstrap an ABM-inferred critical R (with 95% CI) at each traced c.
# --------------------------------------------------------------------------


def infer_abm_critical_points(
    critical_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    primary = [row for row in threshold_rows if row["dataset"] == "CP_THRESHOLD_R1" and row["phase"] == "PRODUCTION"]
    critical_c = np.asarray([row["c_star"] for row in critical_rows])
    critical_R = np.asarray([row["R"] for row in critical_rows])
    inferred: list[dict[str, Any]] = []
    for c_value in sorted({row["c"] for row in primary}):
        cells = sorted((row for row in primary if row["c"] == c_value), key=lambda row: row["R"])
        if len(cells) != 2:
            raise RuntimeError(f"Expected two primary threshold cells at c={c_value:g}")
        R_low, R_high = (row["R"] for row in cells)
        growth_low, growth_high = (row["event_growth_estimate"] for row in cells)
        if not growth_low < 0.0 < growth_high:
            raise RuntimeError(f"ABM growth estimates do not bracket zero at c={c_value:g}")
        inferred_R = R_low - growth_low * (R_high - R_low) / (growth_high - growth_low)
        exact_critical_R = float(np.interp(c_value, critical_c, critical_R))
        inferred.append(
            {
                "c": c_value,
                "lower_bracketing_R": R_low,
                "upper_bracketing_R": R_high,
                "lower_ABM_growth": growth_low,
                "upper_ABM_growth": growth_high,
                "analytical_critical_R": exact_critical_R,
                "inferred_critical_R": inferred_R,
                "bootstrap_95_low": float("nan"),
                "bootstrap_95_high": float("nan"),
            }
        )

    return inferred


def bootstrap_critical_points(
    inferred: list[dict[str, Any]], results: list[Result]
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[float, float], list[Result]] = {}
    for result in results:
        task = result.task
        if task.dataset == "CP_THRESHOLD_R1" and task.phase == "PRODUCTION":
            by_cell.setdefault((task.R, task.c), []).append(result)

    updated: list[dict[str, Any]] = []
    for row in inferred:
        c_value = row["c"]
        R_low, R_high = row["lower_bracketing_R"], row["upper_bracketing_R"]
        cell_low = by_cell[(R_low, c_value)]
        cell_high = by_cell[(R_high, c_value)]
        net_low = np.asarray([item.infection_events - item.primary_removal_events - item.traced_removals for item in cell_low], dtype=float)
        time_low = np.asarray([item.parent_person_time for item in cell_low])
        net_high = np.asarray([item.infection_events - item.primary_removal_events - item.traced_removals for item in cell_high], dtype=float)
        time_high = np.asarray([item.parent_person_time for item in cell_high])

        rng = np.random.default_rng(CRITICAL_POINT_BOOTSTRAP_SEED + int(round(1000 * c_value)))
        bootstrap_R = np.empty(BOOTSTRAPS)
        for start in range(0, BOOTSTRAPS, 250):
            count = min(250, BOOTSTRAPS - start)
            indices_low = rng.integers(0, len(net_low), size=(count, len(net_low)))
            indices_high = rng.integers(0, len(net_high), size=(count, len(net_high)))
            growth_low = net_low[indices_low].sum(axis=1) / time_low[indices_low].sum(axis=1)
            growth_high = net_high[indices_high].sum(axis=1) / time_high[indices_high].sum(axis=1)
            denominator = growth_high - growth_low
            if np.any(denominator <= 0.0):
                raise RuntimeError(f"Nonpositive bootstrap growth secant at c={c_value:g}")
            bootstrap_R[start : start + count] = R_low - growth_low * (R_high - R_low) / denominator
        low, high = np.quantile(bootstrap_R, (0.025, 0.975))
        updated.append({**row, "bootstrap_95_low": float(low), "bootstrap_95_high": float(high)})
    return updated


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-points", type=int, default=201)
    parser.add_argument("--critical-points", type=int, default=241)
    parser.add_argument("--cohort-replicates", type=int, default=300)
    parser.add_argument("--threshold-replicates", type=int, default=300)
    parser.add_argument("--sensitivity-replicates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--smoke", action="store_true", help="small replicate counts and coarse grids for a fast check")
    args = parser.parse_args()
    if args.smoke:
        args.c_points = 41
        args.critical_points = 41
        args.cohort_replicates = 10
        args.threshold_replicates = 40
        args.sensitivity_replicates = 4
    return args


def main() -> None:
    args = parse_arguments()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    analytical_config = AnalyticalConfig(c_points=args.c_points, critical_points=args.critical_points)
    analytical_rows, critical_rows, diagnostics = generate_tables(analytical_config)
    analytical = analytical_lookup(analytical_rows)
    critical_R = critical_R_interpolator(critical_rows)

    tasks = build_tasks(
        analytical,
        critical_R,
        cohort_replicates=args.cohort_replicates,
        threshold_replicates=args.threshold_replicates,
        sensitivity_replicates=args.sensitivity_replicates,
    )
    results = run_tasks(tasks, args.workers)
    replicate_rows = [result_row(result) for result in results]
    threshold_rows, cohort_rows = summarize(results, analytical)

    if not all(int(row["incomplete_cohort"]) == 0 for row in cohort_rows):
        raise RuntimeError("A cohort cell has incomplete follow-up.")
    if not all(not math.isfinite(result.cap_time) for result in results):
        raise RuntimeError("A replicate hit the maximum-active-population cap.")
    production_threshold = [row for row in threshold_rows if row["phase"] == "PRODUCTION" and row["dataset"] == "CP_THRESHOLD_R1"]
    sign_ok = all(
        (row["event_growth_estimate"] < 0.0 if row["boundary_side"] == "below" else row["event_growth_estimate"] > 0.0)
        for row in production_threshold
    )
    interval_ok = all(
        (row["bootstrap_95_high"] < 0.0 if row["boundary_side"] == "below" else row["bootstrap_95_low"] > 0.0)
        for row in production_threshold
    )
    if not sign_ok:
        raise RuntimeError("A production threshold cell did not resolve the predicted growth sign.")
    if not interval_ok:
        raise RuntimeError("A production threshold cell's 95% CI did not resolve the predicted sign.")

    inferred_critical = infer_abm_critical_points(critical_rows, threshold_rows)
    inferred_critical = bootstrap_critical_points(inferred_critical, results)

    write_csv(DATA_DIR / "analytical_curves.csv", analytical_rows)
    write_csv(DATA_DIR / "critical_curve.csv", critical_rows)
    write_csv(DATA_DIR / "abm_replicates.csv", replicate_rows)
    write_csv(DATA_DIR / "abm_threshold_summary.csv", threshold_rows)
    write_csv(DATA_DIR / "abm_cohort_summary.csv", cohort_rows)
    write_csv(DATA_DIR / "abm_inferred_critical_points.csv", inferred_critical)

    manifest = {
        "schema_version": "1.0",
        "R_curve_values": list(R_CURVES),
        "solver": asdict(analytical_config),
        "diagnostics": diagnostics,
        "replicate_rows": len(replicate_rows),
        "cohort_summary_rows": len(cohort_rows),
        "threshold_summary_rows": len(threshold_rows),
        "maximum_relative_cohort_MCSE": max(row["relative_MCSE"] for row in cohort_rows),
        "all_cohorts_complete": True,
        "zero_cap_censoring": True,
        "threshold_sign_agreement": sign_ok,
        "threshold_intervals_resolve_predicted_sign": interval_ok,
        "sources": {"canonical_qss.py": file_sha256(HERE / "canonical_qss.py"), "figure3_measurement_windows.json": file_sha256(WINDOWS_PATH)},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"Generated Figure 5 data: {len(analytical_rows)} analytical rows, "
        f"{len(critical_rows)} critical-curve rows, {len(tasks)} ABM paths; "
        f"R_*(c=1)={diagnostics['critical_endpoint_R_at_c1']:.10f}; "
        f"max cohort relative MCSE={manifest['maximum_relative_cohort_MCSE']:.4%}; "
        f"threshold sign agreement={sign_ok}, intervals resolve sign={interval_ok}."
    )


if __name__ == "__main__":
    main()
