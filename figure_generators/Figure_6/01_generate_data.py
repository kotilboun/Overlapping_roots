#!/usr/bin/env python3
"""Step 1/2 for Figure 6: simulate the finite-pool ABM and cache closure terms.

Runs the finite-susceptible-pool event-driven ABM (120 independent
realizations per (R, c) cell over R in {1,1.5,2,4}, c in {0,0.25,0.5,0.75,1}),
averages the genealogical moments over the ensemble at each observation time,
and evaluates the zeroth-order algebraic QSS closure and the depth-K=1,2,3
dynamic tail closures at that ensemble-mean state. Pure data generation -- no
matplotlib import. Writes every numeric result to ``data/``.

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
from typing import Any

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

R_VALUES = (1.0, 1.5, 2.0, 4.0)
C_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
CLOSURES = ("algebraic_qss0", "dynamic_K1", "dynamic_K2", "dynamic_K3")


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

    def moments(self, kmax: int) -> np.ndarray:
        n_nodes = len(self.parent)
        active = np.asarray(self.infectious, dtype=float)
        out = np.zeros(kmax + 1, dtype=float)
        out[0] = active.sum()
        if out[0] == 0.0:
            return out
        parents = np.asarray(self.parent, dtype=np.int64)
        has_parent = parents >= 0
        path_counts = active
        for depth in range(1, kmax + 1):
            child_sum = np.bincount(
                parents[has_parent],
                weights=path_counts[has_parent],
                minlength=n_nodes,
            )
            path_counts = active * child_sum
            out[depth] = path_counts.sum()
        return out


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
        observed = tree.moments(4)
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


def simulate_finite_pool(
    R: float,
    c: float,
    protocol: Protocol,
    seed: np.random.SeedSequence,
) -> np.ndarray:
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
        payload[f"R{R:g}_c{c:g}".replace(".", "p")] = trajectories
    return payload


def conditional_moment_mean(trajectories: np.ndarray, k: int) -> np.ndarray:
    I = trajectories[:, :, 1]
    Mk = trajectories[:, :, k + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(I > 0.0, Mk / I, np.nan)
    valid_count = np.sum(np.isfinite(ratios), axis=0)
    summed = np.nansum(ratios, axis=0)
    return np.divide(summed, valid_count, out=np.full_like(summed, np.nan), where=valid_count > 0)


def tail_closure(R_U: np.ndarray, K: int, mK: np.ndarray) -> np.ndarray:
    return np.asarray(mK) * np.asarray(R_U) / (np.asarray(R_U) + K + 1.0)


def m10(R_U: np.ndarray) -> np.ndarray:
    R_U = np.maximum(np.asarray(R_U), 0.0)
    return R_U / (R_U + 1.0)


def build_ensemble_terms(
    payload: dict[str, np.ndarray], protocol: Protocol
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    times = payload["times"]
    post = times >= protocol.tau_on
    buckets: dict[str, dict[str, list[np.ndarray]]] = {
        closure: {name: [] for name in ("x", "y", "tau", "R", "c")} for closure in CLOSURES
    }
    summaries: list[dict[str, Any]] = []

    for R in R_VALUES:
        for c in C_VALUES:
            key = f"R{R:g}_c{c:g}".replace(".", "p")
            tr = payload[key]
            U_bar = tr[:, :, 0].mean(axis=0)
            I_bar = tr[:, :, 1].mean(axis=0)
            mbar = {k: conditional_moment_mean(tr, k) for k in range(1, 5)}
            R_U = R * U_bar / protocol.U0

            panel_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            panel_values["algebraic_qss0"] = (c * m10(R_U), c * mbar[1])
            for K in (1, 2, 3):
                phi = tail_closure(R_U, K, mbar[K])
                x = c * (mbar[1] * mbar[K] - phi)
                y = c * (mbar[1] * mbar[K] - mbar[K + 1])
                panel_values[f"dynamic_K{K}"] = (x, y)

            for closure, (x_all, y_all) in panel_values.items():
                valid = post & np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(I_bar)
                x = x_all[valid]
                y = y_all[valid]
                tau = times[valid]
                buckets[closure]["x"].append(x)
                buckets[closure]["y"].append(y)
                buckets[closure]["tau"].append(tau)
                buckets[closure]["R"].append(np.full(len(x), R))
                buckets[closure]["c"].append(np.full(len(x), c))
                residual = x - y
                summaries.append(
                    {
                        "closure": closure,
                        "R": R,
                        "c": c,
                        "n_replicates": protocol.replicates,
                        "n_timepoints": int(len(x)),
                        "mean_signed_defect": float(np.mean(residual)) if len(residual) else np.nan,
                        "mean_absolute_defect": float(np.mean(np.abs(residual))) if len(residual) else np.nan,
                        "root_mean_square_defect": float(np.sqrt(np.mean(residual**2))) if len(residual) else np.nan,
                        "maximum_absolute_defect": float(np.max(np.abs(residual))) if len(residual) else np.nan,
                    }
                )

    terms: dict[str, dict[str, np.ndarray]] = {}
    for closure, values in buckets.items():
        terms[closure] = {name: np.concatenate(parts) if parts else np.array([]) for name, parts in values.items()}
    return terms, summaries


def round_two_significant_digits(value: float) -> float:
    """Round a nonzero value to two significant digits."""
    if not np.isfinite(value) or value == 0.0:
        return float(value)
    decimals = int(1 - np.floor(np.log10(abs(value))))
    return float(np.round(value, decimals))


def histogram_statistics(terms: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for closure in CLOSURES:
        data = terms[closure]
        delta = data["x"] - data["y"]
        selected = np.isfinite(delta) & (data["c"] > 0.0)
        values = delta[selected]
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=0))
        ticks = [
            round_two_significant_digits(mean - 2.0 * standard_deviation),
            round_two_significant_digits(mean),
            round_two_significant_digits(mean + 2.0 * standard_deviation),
        ]
        rows.append(
            {
                "closure": closure,
                "delta_definition": "closure_supplied_minus_ABM_measured",
                "n_snapshot_values": len(values),
                "mean": mean,
                "standard_deviation": standard_deviation,
                "tick_mean_minus_2sd": ticks[0],
                "tick_mean": ticks[1],
                "tick_mean_plus_2sd": ticks[2],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_risk_rows(payload: dict[str, np.ndarray], protocol: Protocol) -> list[dict[str, Any]]:
    risk_rows: list[dict[str, Any]] = []
    for R in R_VALUES:
        for c in C_VALUES:
            key = f"R{R:g}_c{c:g}".replace(".", "p")
            infectious = payload[key][:, :, 1]
            for time_index, tau in enumerate(payload["times"]):
                risk_rows.append(
                    {
                        "R": R,
                        "c": c,
                        "tau": float(tau),
                        "requested": protocol.replicates,
                        "alive": int(np.sum(infectious[:, time_index] > 0.0)),
                    }
                )
    return risk_rows


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
        protocol = Protocol(replicates=4, U0=400, I0=20, tau_on=0.5, tau_end=1.2, num_times=37, seed=20260815, workers=2)
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

    start = time.perf_counter()
    payload = simulate_all(protocol)
    terms, summaries = build_ensemble_terms(payload, protocol)
    elapsed = time.perf_counter() - start

    np.savez_compressed(DATA_DIR / "trajectories.npz", **payload)
    term_payload = {f"{closure}_{name}": arr for closure, vals in terms.items() for name, arr in vals.items()}
    np.savez_compressed(DATA_DIR / "snapshot_terms.npz", **term_payload)
    write_csv(DATA_DIR / "closure_summary.csv", summaries)

    risk_rows = build_risk_rows(payload, protocol)
    write_csv(DATA_DIR / "alive_risk_sets.csv", risk_rows)
    minimum_alive_post_activation = min(
        int(row["alive"]) for row in risk_rows if float(row["tau"]) >= protocol.tau_on
    )

    histogram_rows = histogram_statistics(terms)
    write_csv(DATA_DIR / "histogram_tick_summary.csv", histogram_rows)

    manifest = {
        "protocol": asdict(protocol),
        "R_values": list(R_VALUES),
        "c_values": list(C_VALUES),
        "moment_counter_validation": "passed",
        "estimand": "closure evaluated at ensemble-mean ABM state",
        "snapshot_error": "closure-supplied minus ABM-measured",
        "histogram_tick_rule": (
            "mean-2 population SD, mean, mean+2 population SD; each rounded to two significant digits"
        ),
        "elapsed_seconds": elapsed,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": __import__("scipy").__version__},
        "minimum_alive_post_activation": minimum_alive_post_activation,
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Generated Figure 6 data: {len(summaries)} closure-summary rows, "
        f"minimum alive-post-activation count {minimum_alive_post_activation}, "
        f"elapsed {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
