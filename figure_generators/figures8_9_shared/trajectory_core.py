"""Shared finite-pool ABM engine and closure solvers for Figures 8 and 9.

Figures 8 and 9 both show representative R=4 finite-pool trajectories
overlaid with deterministic closures (Sect. 7.7): Figure 8 overlays the
depth-K dynamic closures (K=1,2,3) and Figure 9 overlays the zeroth-order
algebraic QSS closure. They read the same underlying ABM ensemble, so this
module holds the one copy of the simulation and closure code that both
folders' ``01_generate_data.py`` import.

The ABM engine, closure ODEs, and RNG seeding are copied unchanged from
``figure_generators/Figure_8/01_generate_data.py`` (same engine, same master
seed 20260815) so that, at the shared parameter cells (R=4), the simulated
trajectories are bit-for-bit identical to the ones underlying Figure 8 --
this is verified in each folder's generator by comparing against Figure 8's
cached ``data/source_abm_trajectories.npz``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

CI_Z = 1.959963984540054  # two-sided 95% normal quantile


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


# ---------------------------------------------------------------------------
# Finite-susceptible-pool ABM (identical engine to Figure 6/Figure 7).
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


def simulate_replicates(R: float, c: float, protocol: Protocol, master_seed: np.random.SeedSequence) -> np.ndarray:
    """Simulate ``protocol.replicates`` independent trajectories for one (R, c) cell."""
    seeds = master_seed.spawn(protocol.replicates)
    trajectories = np.zeros((protocol.replicates, protocol.num_times, 6), dtype=float)
    for rep, seed in enumerate(seeds):
        trajectories[rep] = simulate_finite_pool(R, c, protocol, seed)
    return trajectories


def cell_seed(master_seed: int, R_values: tuple[float, ...], c_values: tuple[float, ...], R: float, c: float) -> np.random.SeedSequence:
    """Deterministic per-(R,c) seed spawned from the master seed, in grid order.

    Matches Figure 7's ``simulate_all``: one SeedSequence is spawned per (R, c)
    pair in the order ``[(R, c) for R in R_values for c in c_values]``, so the
    cell for a given (R, c) reproduces Figure 7's trajectories bit-for-bit.
    """
    pairs = [(rv, cv) for rv in R_values for cv in c_values]
    index = pairs.index((R, c))
    master = np.random.SeedSequence(master_seed)
    return master.spawn(len(pairs))[index]


# ---------------------------------------------------------------------------
# Deterministic closures (Eqs. 7.1-7.6), independent of the ABM.
# ---------------------------------------------------------------------------


def m10(R_U: np.ndarray | float) -> np.ndarray | float:
    """Zeroth-order algebraic QSS closure, Eq. (7.6): m_1^(0)(R_U) = R_U/(R_U+1)."""
    return np.asarray(R_U) / (np.asarray(R_U) + 1.0)


def tail(R_U: float, K: int, mK: float) -> float:
    """Depth-K dynamic tail map, Eq. (7.1)."""
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
) -> np.ndarray:
    """Zeroth-order algebraic QSS closure trajectory, Eqs. (7.6)-(7.7)."""
    def rhs(active_c: float) -> Callable[[float, np.ndarray], np.ndarray]:
        def evaluate(_tau: float, state: np.ndarray) -> np.ndarray:
            U, I = state
            R_U = R * U / U0
            incidence = R_U * I
            tracing = active_c * I * float(m10(R_U))
            return np.asarray((-incidence, incidence - I - tracing))

        return evaluate

    state = solve_piecewise(np.asarray((U0, I0)), times, rhs(0.0), rhs(c), max_step, tau_on)
    return state  # columns: U, I


def solve_dynamic(
    R: float, c: float, K: int, times: np.ndarray, max_step: float, U0: float, I0: float, tau_on: float
) -> np.ndarray:
    """Depth-K dynamic closure trajectory (Sect. 7.1)."""
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
    return state  # columns: U, I, m1, ..., mK


# ---------------------------------------------------------------------------
# Post-processing: pointwise ABM summaries, m1/flux, and trajectory metrics.
# ---------------------------------------------------------------------------


def pointwise_summary(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Ensemble mean and 2.5th-97.5th replicate percentile band, over time.

    ``counts`` has shape (replicates, num_times); this is a plain empirical
    percentile of the 120 replicates at each time, not a bootstrap estimate.
    """
    return {
        "mean": counts.mean(axis=0),
        "q025": np.percentile(counts, 2.5, axis=0),
        "q975": np.percentile(counts, 97.5, axis=0),
    }


def i_u_m1_flux(raw: np.ndarray, c: float, U0: float) -> dict[str, np.ndarray]:
    """Derive the four plotted coordinates from raw ABM columns U, I, M1, ...

    ``raw`` has shape (..., >=3): columns U, I, M1 (extra moment columns, if
    present, are ignored). Returns i=I/U0, u=U/U0, m1=M1/I (0 where I=0), and
    flux=c*m1*i.
    """
    U, I, M1 = raw[..., 0], raw[..., 1], raw[..., 2]
    i = I / U0
    u = U / U0
    m1 = np.divide(M1, I, out=np.zeros_like(M1), where=I > 0)
    flux = c * m1 * i
    return {"i": i, "u": u, "m1": m1, "flux": flux}


def closure_i_u_m1_flux(state: np.ndarray, c: float, U0: float) -> dict[str, np.ndarray]:
    """Same four coordinates from a closure's (U, I, m1, ...) state trajectory."""
    U, I, m1 = state[:, 0], state[:, 1], state[:, 2]
    i = I / U0
    u = U / U0
    flux = c * m1 * i
    return {"i": i, "u": u, "m1": m1, "flux": flux}


def trajectory_error(closure_z: np.ndarray, abm_z: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Normalized integrated L2 error between a closure and the ABM mean, Eq. (7.16)."""
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


def delta_method_interval(raw_z: np.ndarray, closure_z: np.ndarray, times: np.ndarray) -> tuple[float, float, float, float]:
    """95% Monte Carlo interval for E_C from complete-realization influence values.

    ``raw_z`` has shape (replicates, num_times, 3); ``closure_z`` has shape
    (num_times, 3). Returns (point_error, standard_error, ci_low, ci_high).
    """
    n_replicates = raw_z.shape[0]
    abm_mean = raw_z.mean(axis=0)
    point = float(trajectory_error(closure_z, abm_mean, times))
    duration = float(times[-1] - times[0])
    weights = trapezoid_weights(times)
    difference = closure_z - abm_mean
    gradient = -(weights[:, None] / duration) * difference / max(point, 1e-15)
    centered_paths = raw_z - abm_mean[None, :, :]
    influence = np.einsum("tj,rtj->r", gradient, centered_paths)
    standard_error = float(influence.std(ddof=1) / np.sqrt(n_replicates))
    low = max(0.0, point - CI_Z * standard_error)
    high = point + CI_Z * standard_error
    return point, standard_error, low, high


def stochastic_spread(raw_z: np.ndarray, times: np.ndarray) -> float:
    """Time-integrated RMS replicate spread around the ABM ensemble mean.

    sqrt of the duration-averaged, replicate-averaged squared deviation from
    the ensemble mean, in the same trajectory-vector norm as trajectory_error.
    """
    abm_mean = raw_z.mean(axis=0)
    deviation = raw_z - abm_mean[None, :, :]
    squared_norm = np.sum(deviation * deviation, axis=-1)  # (replicates, num_times)
    per_replicate = squared_norm.mean(axis=0)  # (num_times,)
    duration = float(times[-1] - times[0])
    return float(np.sqrt(np.trapezoid(per_replicate, x=times) / duration))
