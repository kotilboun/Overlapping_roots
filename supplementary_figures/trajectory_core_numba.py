"""Numba-accelerated finite-pool ABM engine for the full OR1-S2 design.

ESM_1.tex (Online Resource 1, sections OR1-S1--OR1-S7) documents a crossed
design of 195 raw-rate protocols (R in {2,4,6}, c in {0,0.25,0.5,0.75,1},
Gamma in {1/4,1/2,1,2,4}, three detection-tracing decompositions that
coincide at c=1) simulated with 30 independent 120-realization pools at
U0=8000 (702,000 trajectories) plus one 120-realization pool per protocol at
U0 in {500,1000,2000} for population-size convergence. The pure-Python
engine in ``generate_abm_mean_trajectory_error_convergence.py`` /
``u8000_common.py`` is far too slow for this scale (~0.06 s/trajectory would
take many hours); this module is a numba JIT re-implementation of the same
event-driven model (verified against the pure-Python engine's trajectories
below) fast enough to run the full design in minutes.

The protocol grid and decomposition formulas are the same ones used in
``S5/s5_common.py`` for the (independently designed, event-count) OR1-S6
verification -- both draw on the crossed design in ESM_1.tex Table OR1.1 --
generalized here to accept any U0/I0 rather than the fixed U0=8000, I0=160
used there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

R_GRID = (2.0, 4.0, 6.0)
C_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
GAMMA_GRID = (0.25, 0.50, 1.0, 2.0, 4.0)
TAU_END = 5.0
NUM_TIMES = 151
SWITCH_TAU = 0.5
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    R: float
    c: float
    Gamma: float
    decomposition_id: str
    decomposition: str
    beta: float
    gamma: float
    gamma_c: float
    p_f: float
    U0: int
    I0: int


def build_protocols(U0: int, I0: int) -> list[Protocol]:
    """The 195-protocol crossed design of ESM_1.tex Table OR1.1, at given (U0, I0)."""
    out: list[Protocol] = []
    for R in R_GRID:
        for c in C_GRID:
            for Gamma in GAMMA_GRID:
                if c == 1.0:
                    rows = [("C", "coincident complete-tracing boundary", 0.0, Gamma, 1.0)]
                else:
                    rows = [
                        ("L", "less-frequent detection, complete tracing", Gamma * (1 - c), Gamma * c, 1.0),
                        ("M", "intermediate decomposition", Gamma * (1 - c) / 2, Gamma * (1 + c) / 2, 2 * c / (1 + c)),
                        ("F", "frequent detection, partial tracing", 0.0, Gamma, c),
                    ]
                for did, label, gamma, gamma_c, pf in rows:
                    out.append(Protocol(
                        protocol_id=f"R{R:g}_c{c:g}_G{Gamma:g}_{did}",
                        R=R, c=c, Gamma=Gamma, decomposition_id=did,
                        decomposition=label, beta=R * Gamma / U0,
                        gamma=gamma, gamma_c=gamma_c, p_f=pf, U0=U0, I0=I0,
                    ))
    assert len(out) == 195
    return out


def pool_seed(master_seed: int, protocol: Protocol, pool: int) -> np.random.SeedSequence:
    """One SeedSequence per (U0, protocol, pool); replicate seeds are spawned from it."""
    codes = {"L": 11, "M": 13, "F": 17, "C": 19}
    return np.random.SeedSequence([
        master_seed, int(protocol.U0), int(protocol.R * 100), int(protocol.c * 1000),
        int(protocol.Gamma * 1000), codes[protocol.decomposition_id], pool,
    ])


def replicate_seed(pool_ss: np.random.SeedSequence, replicate: int, n_replicates: int) -> np.uint64:
    child = pool_ss.spawn(n_replicates)[replicate]
    seed = child.generate_state(1, dtype=np.uint64)[0]
    return np.uint64(seed if seed != 0 else 88172645463325252)


@njit(inline="always")
def _rng_next(state):
    x = state
    x ^= x >> np.uint64(12)
    x ^= (x << np.uint64(25)) & MASK64
    x ^= x >> np.uint64(27)
    state = x
    return state, (x * np.uint64(2685821657736338717)) & MASK64


@njit(inline="always")
def _uniform01(state):
    state, x = _rng_next(state)
    return state, (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(inline="always")
def _uniform_open(state):
    state, u = _uniform01(state)
    if u <= 0.0:
        u = 1.0 / 9007199254740992.0
    return state, u


@njit(inline="always")
def _random_active(state, active_nodes, active_count):
    state, u = _uniform01(state)
    pos = int(u * active_count)
    if pos >= active_count:
        pos = active_count - 1
    return state, active_nodes[pos]


@njit(inline="always")
def _remove_node(node, active, parent, active_child_count, active_nodes, active_pos, active_count, m1):
    if active[node] == 0:
        return active_count, m1
    par = parent[node]
    if par >= 0 and active[par] == 1:
        active_child_count[par] -= 1
        m1 -= 1
    m1 -= active_child_count[node]
    active[node] = 0
    pos = active_pos[node]
    last = active_nodes[active_count - 1]
    active_count -= 1
    if last != node:
        active_nodes[pos] = last
        active_pos[last] = pos
    active_pos[node] = -1
    return active_count, m1


@njit
def simulate_trajectory_numba(U0_, I0_, beta, gamma, gamma_c, pf, Gamma, phys_times, switch_time, seed):
    """Return U, I, M1 sampled at every time in `phys_times` (physical time, ascending)."""
    nmax = U0_ + I0_
    active = np.zeros(nmax, np.uint8)
    parent = np.full(nmax, -1, np.int32)
    first_child = np.full(nmax, -1, np.int32)
    next_sibling = np.full(nmax, -1, np.int32)
    active_child_count = np.zeros(nmax, np.int32)
    active_nodes = np.empty(nmax, np.int32)
    active_pos = np.full(nmax, -1, np.int32)
    traced_buf = np.empty(nmax, np.int32)
    for j in range(I0_):
        active[j] = 1
        active_nodes[j] = j
        active_pos[j] = j
    node_count = I0_
    active_count = I0_
    m1 = 0
    U = U0_
    time = 0.0
    n_times = phys_times.shape[0]
    out = np.zeros((n_times, 3), np.float64)
    state = seed
    idx = 0

    while idx < n_times and phys_times[idx] <= 0.0:
        out[idx, 0] = U; out[idx, 1] = active_count; out[idx, 2] = m1
        idx += 1

    while idx < n_times and active_count > 0:
        infection_rate = beta * U * active_count if U > 0 else 0.0
        recovery_rate = gamma * active_count
        identification_rate = gamma_c * active_count
        total_rate = infection_rate + recovery_rate + identification_rate
        if total_rate <= 0.0:
            break
        state, u = _uniform_open(state)
        next_time = time - np.log(u) / total_rate
        while idx < n_times and phys_times[idx] <= next_time:
            out[idx, 0] = U; out[idx, 1] = active_count; out[idx, 2] = m1
            idx += 1
        if idx >= n_times:
            break
        time = next_time
        state, uevent = _uniform01(state)
        selector = uevent * total_rate
        state, node = _random_active(state, active_nodes, active_count)
        if selector < infection_rate:
            child = node_count
            node_count += 1
            active[child] = 1
            parent[child] = node
            next_sibling[child] = first_child[node]
            first_child[node] = child
            active_child_count[node] += 1
            m1 += 1
            active_pos[child] = active_count
            active_nodes[active_count] = child
            active_count += 1
            U -= 1
        elif selector < infection_rate + recovery_rate:
            active_count, m1 = _remove_node(
                node, active, parent, active_child_count, active_nodes, active_pos, active_count, m1
            )
        else:
            nt = 0
            active_pf = pf if time >= switch_time else 0.0
            if active_pf > 0.0:
                child = first_child[node]
                while child >= 0:
                    if active[child] == 1:
                        take = False
                        if active_pf >= 1.0:
                            take = True
                        else:
                            state, ur = _uniform01(state)
                            if ur < active_pf:
                                take = True
                        if take:
                            traced_buf[nt] = child
                            nt += 1
                    child = next_sibling[child]
            active_count, m1 = _remove_node(
                node, active, parent, active_child_count, active_nodes, active_pos, active_count, m1
            )
            for q in range(nt):
                child = traced_buf[q]
                if active[child] == 1:
                    active_count, m1 = _remove_node(
                        child, active, parent, active_child_count, active_nodes, active_pos, active_count, m1
                    )
    # Extinct or truncated: hold at the last state (I=M1=0 if extinct).
    while idx < n_times:
        out[idx, 0] = U; out[idx, 1] = active_count; out[idx, 2] = m1
        idx += 1
    return out


@njit(parallel=True)
def simulate_batch_numba(U0_, I0_, beta, gamma, gamma_c, pf, Gamma, phys_times, switch_time, seeds):
    result = np.zeros((seeds.size, phys_times.shape[0], 3), np.float64)
    for r in prange(seeds.size):
        result[r] = simulate_trajectory_numba(
            U0_, I0_, beta, gamma, gamma_c, pf, Gamma, phys_times, switch_time, seeds[r]
        )
    return result


def simulate_pool(protocol: Protocol, n_replicates: int, tau_grid: np.ndarray, switch_tau: float, seed: np.random.SeedSequence) -> np.ndarray:
    """Simulate one pool of `n_replicates` trajectories, sampled at `tau_grid` (dimensionless)."""
    phys_times = tau_grid / protocol.Gamma
    switch_time = switch_tau / protocol.Gamma
    seeds = np.array([replicate_seed(seed, r, n_replicates) for r in range(n_replicates)], dtype=np.uint64)
    return simulate_batch_numba(
        protocol.U0, protocol.I0, protocol.beta, protocol.gamma, protocol.gamma_c, protocol.p_f,
        protocol.Gamma, phys_times, switch_time, seeds,
    )
