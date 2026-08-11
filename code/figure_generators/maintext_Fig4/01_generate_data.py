#!/usr/bin/env python3
"""Step 1/2 for Figure 7: simulate the ABM and solve the closure trajectories.

Re-runs the identical finite-susceptible-pool event-driven ABM used for
Figure 6 (same engine, same seed 20260815) so this folder is self-contained
and does not depend on Figure 6's cached data. Solves the zeroth-order
algebraic QSS closure and the depth-K=1,2,3 dynamic tail closures with a
switch-restarted ODE integrator, then computes the normalized integrated L2
trajectory error E_C (Eq. 7.16) between each closure and the ensemble-mean
ABM trajectory, with whole-realization influence-function/delta-method 95%
Monte Carlo intervals. Pure data generation -- no matplotlib import. Writes
every numeric result to ``data/``.

Run:
    python 01_generate_data.py --workers 8

A fast end-to-end check is available with:
    python 01_generate_data.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
from scipy import sparse
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

R_VALUES = (1.0, 1.5, 2.0, 4.0)
C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
CLOSURES = ("algebraic_qss0", "dynamic_K1", "dynamic_K2", "dynamic_K3")
CI_Z = 1.959963984540054


@dataclass
class Protocol:
    replicates: int = 120
    U0: int = 8000
    I0: int = 160
    tau_on: float = 0.5
    tau_end: float = 5.0
    num_times: int = 151
    gamma: float = 0.0
    gamma_c: float = 1.0
    seed: int = 20260815
    workers: int = 1


def archive_key(R: float, c: float) -> str:
    return f"R{R:g}_c{c:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# Part 1: finite-susceptible-pool ABM. Identical engine and seed to Figure 6's
# 01_generate_data.py, so the resulting trajectories are bit-for-bit the same
# archive used there (verified by SHA-256 during development).
# ---------------------------------------------------------------------------


class ActiveGenealogy:
    """Active transmission forest with exact descendant-depth counting."""

    def __init__(self, initial_infectious: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.parent: list[int] = []
        self.children: list[list[int]] = []
        self.infectious: list[bool] = []
        self.infectious_nodes: list[int] = []
        self.infectious_position: dict[int, int] = {}
        self.birth_time: list[float] = []
        self.removal_time: list[float] = []
        for _ in range(initial_infectious):
            self._append(-1, 0.0)

    def _append(self, parent: int, birth_time: float = 0.0) -> int:
        node = len(self.parent)
        self.parent.append(parent)
        self.children.append([])
        self.infectious.append(True)
        self.infectious_position[node] = len(self.infectious_nodes)
        self.infectious_nodes.append(node)
        self.birth_time.append(float(birth_time))
        self.removal_time.append(np.inf)
        return node

    def random_infectious(self) -> int:
        return self.infectious_nodes[int(self.rng.integers(len(self.infectious_nodes)))]

    def add_infectee(self, infector: int, birth_time: float = 0.0) -> None:
        infectee = self._append(infector, birth_time)
        self.children[infector].append(infectee)

    def remove(self, node: int, removal_time: float = 0.0) -> None:
        if not self.infectious[node]:
            return
        self.infectious[node] = False
        self.removal_time[node] = float(removal_time)
        position = self.infectious_position.pop(node)
        last = self.infectious_nodes.pop()
        if last != node:
            self.infectious_nodes[position] = last
            self.infectious_position[last] = position

    def detect(self, index_case: int, p_f: float, removal_time: float = 0.0) -> None:
        selected = [index_case]
        if p_f > 0.0:
            selected.extend(
                child
                for child in self.children[index_case]
                if self.infectious[child] and self.rng.random() < p_f
            )
        for node in selected:
            self.remove(node, removal_time)


def exhaustive_moments(tree: ActiveGenealogy, kmax: int) -> np.ndarray:
    out = np.zeros(kmax + 1, dtype=float)
    roots = [i for i, active in enumerate(tree.infectious) if active]
    out[0] = len(roots)
    frontier = roots
    for depth in range(1, kmax + 1):
        nxt: list[int] = []
        for node in frontier:
            nxt.extend(child for child in tree.children[node] if tree.infectious[child])
        out[depth] = len(nxt)
        frontier = nxt
    return out


def _moments(tree: ActiveGenealogy, kmax: int) -> np.ndarray:
    n_nodes = len(tree.parent)
    active = np.asarray(tree.infectious, dtype=float)
    out = np.zeros(kmax + 1, dtype=float)
    out[0] = active.sum()
    if out[0] == 0.0:
        return out
    parents = np.asarray(tree.parent, dtype=np.int64)
    has_parent = parents >= 0
    path_counts = active
    for depth in range(1, kmax + 1):
        child_sum = np.bincount(parents[has_parent], weights=path_counts[has_parent], minlength=n_nodes)
        path_counts = active * child_sum
        out[depth] = path_counts.sum()
    return out


def validate_moment_counter() -> None:
    rng = np.random.default_rng(845731)
    tree = ActiveGenealogy(8, rng)
    for _ in range(80):
        if not tree.infectious_nodes:
            break
        if rng.random() < 0.72:
            tree.add_infectee(tree.random_infectious())
        else:
            tree.detect(tree.random_infectious(), float(rng.uniform()))
        observed = _moments(tree, 4)
        expected = exhaustive_moments(tree, 4)
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"Moment counter failed: {observed} != {expected}")


def trajectory_from_intervals(tree: ActiveGenealogy, U0: int, I0: int, times: np.ndarray) -> np.ndarray:
    """Compute U, I, M1,...,M4 at all times in one vectorized pass."""
    births = np.asarray(tree.birth_time, dtype=float)
    removals = np.asarray(tree.removal_time, dtype=float)
    parents = np.asarray(tree.parent, dtype=np.int64)
    active = (births[:, None] <= times[None, :]) & (times[None, :] < removals[:, None])

    out = np.zeros((len(times), 6), dtype=float)
    transmission_births = np.sort(births[I0:])
    out[:, 0] = U0 - np.searchsorted(transmission_births, times, side="right")
    path_counts = active.astype(np.int16, copy=False)
    out[:, 1] = path_counts.sum(axis=0)
    has_parent = parents >= 0
    child_idx = np.nonzero(has_parent)[0]
    parent_idx = parents[has_parent]
    adjacency = sparse.csr_matrix(
        (np.ones(len(child_idx), dtype=np.int8), (parent_idx, child_idx)),
        shape=(len(parents), len(parents)),
    )
    for depth in range(1, 5):
        child_sum = adjacency @ path_counts
        path_counts = active * child_sum
        out[:, depth + 1] = np.asarray(path_counts.sum(axis=0)).ravel()
    return out


def simulate_finite_pool(R: float, c: float, protocol: Protocol, seed: np.random.SeedSequence) -> np.ndarray:
    """Return columns U, M0=I, M1, ..., M4 on the common tau grid."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, protocol.tau_end, protocol.num_times)
    tree = ActiveGenealogy(protocol.I0, rng)
    U = protocol.U0
    tau = 0.0
    beta = R / protocol.U0

    while tau < protocol.tau_end:
        I = len(tree.infectious_nodes)
        if I == 0:
            break
        transmission = beta * U * I
        spontaneous = protocol.gamma * I
        detection = protocol.gamma_c * I
        total = transmission + spontaneous + detection
        next_tau = tau + float(rng.exponential(1.0 / total))
        if next_tau > protocol.tau_end:
            break
        tau = next_tau
        draw = rng.random() * total
        if U > 0 and draw < transmission:
            tree.add_infectee(tree.random_infectious(), tau)
            U -= 1
        elif draw < transmission + spontaneous:
            tree.remove(tree.random_infectious(), tau)
        else:
            p_f = 0.0 if tau < protocol.tau_on else c
            tree.detect(tree.random_infectious(), p_f, tau)

    return trajectory_from_intervals(tree, protocol.U0, protocol.I0, times)


def _simulate_pair(args: tuple[float, float, Protocol, list[np.random.SeedSequence]]) -> tuple[float, float, np.ndarray]:
    R, c, protocol, seeds = args
    trajectories = np.zeros((protocol.replicates, protocol.num_times, 6), dtype=float)
    for rep, seed in enumerate(seeds):
        trajectories[rep] = simulate_finite_pool(R, c, protocol, seed)
    return R, c, trajectories


def simulate_all(protocol: Protocol) -> dict[str, np.ndarray]:
    pairs = [(R, c) for R in R_VALUES for c in C_VALUES]
    master = np.random.SeedSequence(protocol.seed)
    pair_sequences = master.spawn(len(pairs))
    tasks = []
    for (R, c), pair_seq in zip(pairs, pair_sequences):
        tasks.append((R, c, protocol, pair_seq.spawn(protocol.replicates)))

    payload: dict[str, np.ndarray] = {
        "R_values": np.asarray(R_VALUES),
        "c_values": np.asarray(C_VALUES),
        "times": np.linspace(0.0, protocol.tau_end, protocol.num_times),
    }
    workers = max(1, min(protocol.workers, len(tasks)))
    if workers == 1:
        results = [_simulate_pair(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_simulate_pair, task): task[:2] for task in tasks}
            for done, future in enumerate(as_completed(futures), start=1):
                R, c, trajectories = future.result()
                results.append((R, c, trajectories))
                print(f"[{done:02d}/{len(tasks)}] R={R:g}, c={c:g}", flush=True)
    for R, c, trajectories in results:
        payload[archive_key(R, c)] = trajectories
    return payload


# ---------------------------------------------------------------------------
# Part 2: deterministic closures (Eqs. 7.15-7.16), independent of the ABM.
# ---------------------------------------------------------------------------


def m10(R_U: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(R_U) / (np.asarray(R_U) + 1.0)


def tail(R_U: float, K: int, mK: float) -> float:
    return mK * R_U / (R_U + K + 1.0)


def solve_piecewise(
    initial: np.ndarray,
    times: np.ndarray,
    rhs_before: Callable[[float, np.ndarray], np.ndarray],
    rhs_after: Callable[[float, np.ndarray], np.ndarray],
    max_step: float,
    tau_on: float,
) -> np.ndarray:
    """Integrate to the switch exactly and restart without altering the state."""
    result = np.empty((len(times), len(initial)), dtype=float)
    pre_mask = times <= tau_on
    pre_times = times[pre_mask]
    pre_eval = np.unique(np.append(pre_times, tau_on))
    pre = solve_ivp(
        rhs_before, (float(times[0]), tau_on), initial,
        method="DOP853", t_eval=pre_eval, rtol=1e-10, atol=1e-12, max_step=max_step,
    )
    if not pre.success:
        raise RuntimeError(pre.message)
    result[pre_mask] = pre.y[:, : len(pre_times)].T

    post_mask = times > tau_on
    if np.any(post_mask):
        post = solve_ivp(
            rhs_after, (tau_on, float(times[-1])), pre.y[:, -1],
            method="DOP853", t_eval=times[post_mask], rtol=1e-10, atol=1e-12, max_step=max_step,
        )
        if not post.success:
            raise RuntimeError(post.message)
        result[post_mask] = post.y.T
    return result


def solve_algebraic(
    R: float, c: float, times: np.ndarray, max_step: float, U0: float, I0: float, tau_on: float
) -> tuple[np.ndarray, np.ndarray]:
    def rhs(active_c: float) -> Callable[[float, np.ndarray], np.ndarray]:
        def evaluate(_tau: float, state: np.ndarray) -> np.ndarray:
            U, I = state
            R_U = R * U / U0
            incidence = R_U * I
            tracing = active_c * I * float(m10(R_U))
            return np.asarray((-incidence, incidence - I - tracing))

        return evaluate

    state = solve_piecewise(np.asarray((U0, I0)), times, rhs(0.0), rhs(c), max_step, tau_on)
    U, I = state.T
    m1 = m10(R * U / U0)
    z = np.column_stack((U / U0, I / U0, I * m1 / U0))
    return z, state


def solve_dynamic(
    R: float, c: float, K: int, times: np.ndarray, max_step: float, U0: float, I0: float, tau_on: float
) -> tuple[np.ndarray, np.ndarray]:
    def rhs(active_c: float) -> Callable[[float, np.ndarray], np.ndarray]:
        def evaluate(_tau: float, state: np.ndarray) -> np.ndarray:
            U, I = state[:2]
            retained = state[2:]
            R_U = R * U / U0
            m1 = retained[0]
            derivative = np.empty_like(state)
            derivative[0] = -R_U * I
            derivative[1] = I * (R_U - 1.0 - active_c * m1)
            for index in range(K):
                k = index + 1
                current = retained[index]
                previous = 1.0 if k == 1 else retained[index - 1]
                following = retained[index + 1] if k < K else tail(R_U, K, current)
                derivative[index + 2] = (
                    R_U * (previous - current) - k * current + active_c * (m1 * current - following)
                )
            return derivative

        return evaluate

    initial = np.zeros(K + 2, dtype=float)
    initial[:2] = (U0, I0)
    state = solve_piecewise(initial, times, rhs(0.0), rhs(c), max_step, tau_on)
    U, I, m1 = state[:, 0], state[:, 1], state[:, 2]
    z = np.column_stack((U / U0, I / U0, I * m1 / U0))
    return z, state


def solve_all_closures(
    R: float, c: float, times: np.ndarray, max_step: float, U0: float, I0: float, tau_on: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    vectors: list[np.ndarray] = []
    raw: dict[str, np.ndarray] = {}
    z, state = solve_algebraic(R, c, times, max_step, U0, I0, tau_on)
    vectors.append(z)
    raw[CLOSURES[0]] = state
    for K in (1, 2, 3):
        z, state = solve_dynamic(R, c, K, times, max_step, U0, I0, tau_on)
        vectors.append(z)
        raw[f"dynamic_K{K}"] = state
    return np.stack(vectors), raw


def trajectory_error(closure_z: np.ndarray, abm_z: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return Eq. (7.16) for one or several closure trajectories."""
    difference = closure_z - abm_z
    squared_norm = np.sum(difference * difference, axis=-1)
    duration = float(times[-1] - times[0])
    return np.sqrt(np.trapezoid(squared_norm, x=times, axis=-1) / duration)


def trapezoid_weights(times: np.ndarray) -> np.ndarray:
    """Weights whose dot product with sampled values equals np.trapezoid."""
    differences = np.diff(times)
    weights = np.empty(len(times), dtype=float)
    weights[0] = differences[0] / 2.0
    weights[-1] = differences[-1] / 2.0
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2.0
    return weights


def delta_method_intervals(
    raw_z: np.ndarray, closure_z: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo uncertainty in E_C from complete-realization influence values."""
    n_replicates = raw_z.shape[0]
    abm_mean = raw_z.mean(axis=0)
    point = trajectory_error(closure_z, abm_mean, times)
    duration = float(times[-1] - times[0])
    weights = trapezoid_weights(times)
    difference = closure_z - abm_mean[None, :, :]
    gradient = -(weights[None, :, None] / duration) * difference / point[:, None, None]
    centered_paths = raw_z - abm_mean[None, :, :]
    influence = np.einsum("ktj,rtj->rk", gradient, centered_paths)
    standard_error = influence.std(axis=0, ddof=1) / np.sqrt(n_replicates)
    low = np.maximum(0.0, point - CI_Z * standard_error)
    high = point + CI_Z * standard_error
    return standard_error, low, high


def physical_diagnostics(raw_states: dict[str, np.ndarray]) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    for name, state in raw_states.items():
        diagnostics[f"{name}_minimum_U"] = float(np.min(state[:, 0]))
        diagnostics[f"{name}_minimum_I"] = float(np.min(state[:, 1]))
        if name.startswith("dynamic"):
            moments = state[:, 2:]
            diagnostics[f"{name}_minimum_moment"] = float(np.min(moments))
            augmented = np.column_stack((np.ones(len(state)), moments))
            diagnostics[f"{name}_maximum_ordering_violation"] = float(
                np.max(np.maximum(np.diff(augmented, axis=1), 0.0))
            )
    return diagnostics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--U0", type=int)
    parser.add_argument("--I0", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def make_protocol(args: argparse.Namespace) -> Protocol:
    protocol = Protocol()
    if args.smoke:
        protocol = Protocol(replicates=4, U0=400, I0=20, tau_on=0.5, tau_end=1.2, num_times=25, seed=20260815, workers=2)
    for field in ("replicates", "U0", "I0", "workers", "seed"):
        value = getattr(args, field)
        if value is not None:
            setattr(protocol, field, value)
    if protocol.I0 <= 0 or protocol.U0 <= 0 or protocol.I0 >= protocol.U0:
        raise ValueError("Require 0 < I0 < U0")
    if not np.isclose(protocol.gamma + protocol.gamma_c, 1.0):
        raise ValueError("This script assumes Gamma=gamma+gamma_c=1")
    return protocol


def main() -> None:
    args = parse_arguments()
    protocol = make_protocol(args)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Validating genealogical moment bookkeeping...", flush=True)
    validate_moment_counter()
    print("Bookkeeping validation passed.", flush=True)

    started = time.perf_counter()
    payload = simulate_all(protocol)
    times = payload["times"]
    U0 = float(protocol.U0)
    I0 = float(protocol.I0)
    tau_on = protocol.tau_on

    if len(times) != protocol.num_times or not np.isclose(times[0], 0.0):
        raise ValueError("Inconsistent observation-time grid.")

    rows: list[dict[str, Any]] = []
    closure_payload: dict[str, np.ndarray] = {
        "times": times,
        "R_values": np.asarray(R_VALUES),
        "c_values": np.asarray(C_VALUES),
    }
    all_physical: dict[str, float] = {}
    maximum_solver_difference = 0.0
    maximum_grid_relative_difference = 0.0
    ci_zero_counts = {closure: 0 for closure in CLOSURES}
    noise_ratios: dict[str, list[float]] = {closure: [] for closure in CLOSURES}
    debiased_zero_counts = {closure: 0 for closure in CLOSURES}

    for R in R_VALUES:
        for c in C_VALUES:
            key = archive_key(R, c)
            raw = payload[key]
            if raw.shape != (protocol.replicates, len(times), 6):
                raise ValueError(f"Unexpected shape for {key}: {raw.shape}")
            raw_z = raw[:, :, :3] / U0
            abm_mean = raw_z.mean(axis=0)

            closure_z, raw_states = solve_all_closures(R, c, times, 1.0 / 60.0, U0, I0, tau_on)
            reference_z, _ = solve_all_closures(R, c, times, 1.0 / 120.0, U0, I0, tau_on)
            maximum_solver_difference = max(
                maximum_solver_difference, float(np.max(np.abs(closure_z - reference_z)))
            )
            point = trajectory_error(closure_z, abm_mean, times)
            standard_error, low, high = delta_method_intervals(raw_z, closure_z, times)
            duration = float(times[-1] - times[0])
            mean_sampling_noise_q = float(
                np.trapezoid(raw_z.var(axis=0, ddof=1).sum(axis=1) / raw_z.shape[0], x=times) / duration
            )

            coarse_index = np.arange(0, len(times), 2)
            if coarse_index[-1] != len(times) - 1:
                coarse_index = np.append(coarse_index, len(times) - 1)
            coarse = trajectory_error(closure_z[:, coarse_index, :], abm_mean[coarse_index], times[coarse_index])
            maximum_grid_relative_difference = max(
                maximum_grid_relative_difference,
                float(np.max(np.abs(coarse - point) / np.maximum(point, 1e-15))),
            )

            closure_payload[f"{key}_abm_mean_z"] = abm_mean
            for index, closure in enumerate(CLOSURES):
                closure_payload[f"{key}_{closure}_z"] = closure_z[index]
                observed_q = float(point[index] ** 2)
                noise_ratios[closure].append(mean_sampling_noise_q / observed_q)
                debiased_zero_counts[closure] += int(observed_q <= mean_sampling_noise_q)
                ci_zero_counts[closure] += int(low[index] == 0.0)
                rows.append(
                    {
                        "closure": closure,
                        "R": R,
                        "c": c,
                        "error": float(point[index]),
                        "standard_error": float(standard_error[index]),
                        "ci_low": float(low[index]),
                        "ci_high": float(high[index]),
                        "ci_reaches_zero": bool(low[index] == 0.0),
                        "n_replicates": raw.shape[0],
                        "n_times": len(times),
                        "tau_start": float(times[0]),
                        "tau_end": float(times[-1]),
                    }
                )
            for name, value in physical_diagnostics(raw_states).items():
                all_physical[f"{key}_{name}"] = value

    np.savez_compressed(DATA_DIR / "source_abm_trajectories.npz", **payload)
    np.savez_compressed(DATA_DIR / "closure_trajectories.npz", **closure_payload)
    write_csv(DATA_DIR / "trajectory_error_summary.csv", rows)

    medians = {
        closure: float(np.median([row["error"] for row in rows if row["closure"] == closure]))
        for closure in CLOSURES
    }
    elapsed = time.perf_counter() - started
    manifest = {
        "status": "trajectory validation based on Sections 7.4-7.6",
        "protocol": asdict(protocol),
        "estimand": "Eq. (7.16), closure trajectory versus the sample ensemble-mean raw-count ABM trajectory",
        "trajectory_vector": ["U/U0", "I/U0", "M1/U0"],
        "integration": "composite trapezoid on the shared observation-time grid",
        "uncertainty": {
            "method": "whole-realization influence-function/delta-method normal interval",
            "confidence_level": 0.95,
            "normal_quantile": CI_Z,
            "lower_endpoint_constrained_to_nonnegative_parameter_space": True,
            "cells_reaching_zero_by_closure": ci_zero_counts,
        },
        "monte_carlo_resolution": {
            "median_sampling_noise_to_observed_squared_error_by_closure": {
                closure: float(np.median(noise_ratios[closure])) for closure in CLOSURES
            },
            "cells_with_nonpositive_variance_corrected_squared_error_by_closure": debiased_zero_counts,
        },
        "median_error_across_protocols": medians,
        "maximum_ode_step_halving_absolute_difference": maximum_solver_difference,
        "maximum_every_other_timepoint_relative_error_difference": maximum_grid_relative_difference,
        "physical_diagnostics": {
            "minimum_U": min(v for k, v in all_physical.items() if k.endswith("minimum_U")),
            "minimum_I": min(v for k, v in all_physical.items() if k.endswith("minimum_I")),
            "minimum_dynamic_moment": min(v for k, v in all_physical.items() if k.endswith("minimum_moment")),
            "maximum_dynamic_ordering_violation": max(
                v for k, v in all_physical.items() if k.endswith("maximum_ordering_violation")
            ),
            "state_clipping": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "elapsed_seconds": elapsed,
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Generated Figure 7 data: {len(rows)} trajectory-error rows, "
        f"maximum ODE step-halving difference {maximum_solver_difference:.3e}, "
        f"maximum grid-coarsening relative difference {maximum_grid_relative_difference:.3e}, "
        f"elapsed {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
