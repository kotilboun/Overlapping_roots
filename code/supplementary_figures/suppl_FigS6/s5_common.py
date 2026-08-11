"""Shared model core for Supplementary Figure S5 (event-term validation).

Extracted from the original monolithic generate_supp_S5_event_term_validation.py
so that data generation and figure generation can be split into two
independent scripts (generate_data_S5.py, generate_figure_S5.py) while
sharing the instrumented event-driven ABM kernel, protocol grid, and
analytical-expectation formulas without duplication.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

R_GRID = (2.0, 4.0, 6.0)
C_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
GAMMA_GRID = (0.25, 0.50, 1.0, 2.0, 4.0)
U0 = 8000
I0 = 160
N_REPLICATES = 120
TAU_END = 5.0
N_BINS = 150
TRACE_ON = 0.5
MASTER_SEED = 20260817
FIG_WIDTH_IN = 6.85
FIG_HEIGHT_IN = 6.15
EVENT_NAMES = ("transmission", "recovery", "identification", "traced_removal")
EVENT_LABELS = (
    r"Transmission: $\int \beta S I\,\mathrm{d}t$",
    r"Spontaneous removal: $\int \gamma I\,\mathrm{d}t$",
    r"Identification: $\int \gamma_c I\,\mathrm{d}t$",
    r"Traced removal: $\int \gamma_c p_f M_1\,\mathrm{d}t$",
)
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
    U0: int = U0
    I0: int = I0


def build_protocols() -> list[Protocol]:
    out: list[Protocol] = []
    for R in R_GRID:
        for c in C_GRID:
            for Gamma in GAMMA_GRID:
                if c == 1.0:
                    rows = [("C", "coincident complete-tracing boundary", 0.0, Gamma, 1.0)]
                else:
                    rows = [
                        ("L", "less-frequent detection, complete tracing", Gamma*(1-c), Gamma*c, 1.0),
                        ("M", "intermediate decomposition", Gamma*(1-c)/2, Gamma*(1+c)/2, 2*c/(1+c)),
                        ("F", "frequent detection, partial tracing", 0.0, Gamma, c),
                    ]
                for did, label, gamma, gamma_c, pf in rows:
                    out.append(Protocol(
                        protocol_id=f"R{R:g}_c{c:g}_G{Gamma:g}_{did}",
                        R=R, c=c, Gamma=Gamma, decomposition_id=did,
                        decomposition=label, beta=R*Gamma/U0,
                        gamma=gamma, gamma_c=gamma_c, p_f=pf,
                    ))
    assert len(out) == 195
    return out


def trajectory_seed(protocol: Protocol, replicate: int, master_seed: int) -> np.uint64:
    codes = {"L": 11, "M": 13, "F": 17, "C": 19}
    ss = np.random.SeedSequence([
        master_seed, int(protocol.R*100), int(protocol.c*1000),
        int(protocol.Gamma*1000), codes[protocol.decomposition_id], replicate,
    ])
    seed = ss.generate_state(1, dtype=np.uint64)[0]
    return np.uint64(seed if seed != 0 else 88172645463325252)


@njit(inline="always")
def rng_next(state):
    x = state
    x ^= x >> np.uint64(12)
    x ^= (x << np.uint64(25)) & MASK64
    x ^= x >> np.uint64(27)
    state = x
    return state, (x * np.uint64(2685821657736338717)) & MASK64


@njit(inline="always")
def uniform01(state):
    state, x = rng_next(state)
    return state, (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(inline="always")
def uniform_open(state):
    state, u = uniform01(state)
    if u <= 0.0:
        u = 1.0 / 9007199254740992.0
    return state, u


@njit(inline="always")
def random_active(state, active_nodes, active_count):
    state, u = uniform01(state)
    pos = int(u * active_count)
    if pos >= active_count:
        pos = active_count - 1
    return state, active_nodes[pos]


@njit(inline="always")
def remove_node(node, active, parent, active_child_count, active_nodes, active_pos, active_count, m1):
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


@njit(inline="always")
def accumulate_exposure(out, tau0, tau1, U, I, m1, tau_end, n_bins, trace_on):
    end = tau1 if tau1 < tau_end else tau_end
    cur = tau0
    dtau = tau_end / n_bins
    while cur < end - 1e-14:
        idx = int(cur / dtau)
        if idx < 0:
            idx = 0
        if idx >= n_bins:
            idx = n_bins - 1
        cut = (idx + 1) * dtau
        if cut > end:
            cut = end
        if cur < trace_on and trace_on < cut:
            cut = trace_on
        width = cut - cur
        if width <= 0.0:
            cur += 1e-14
            continue
        out[idx, 4] += U * I * width
        out[idx, 5] += I * width
        if cur >= trace_on - 1e-14:
            out[idx, 6] += m1 * width
        cur = cut


@njit
def simulate_one_numba(U0_, I0_, beta, gamma, gamma_c, pf, Gamma, tau_end, n_bins, trace_on, seed):
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
    tau = 0.0
    out = np.zeros((n_bins, 7), np.float64)
    state = seed
    dtau = tau_end / n_bins

    while tau < tau_end - 1e-14 and active_count > 0:
        infection_rate = beta * U * active_count if U > 0 else 0.0
        recovery_rate = gamma * active_count
        identification_rate = gamma_c * active_count
        total_rate = infection_rate + recovery_rate + identification_rate
        if total_rate <= 0.0:
            break
        state, u = uniform_open(state)
        next_time = time - math.log(u) / total_rate
        next_tau = Gamma * next_time
        accumulate_exposure(out, tau, next_tau, U, active_count, m1, tau_end, n_bins, trace_on)
        if next_tau >= tau_end:
            break
        event_bin = int(next_tau / dtau)
        if event_bin >= n_bins:
            event_bin = n_bins - 1
        state, uevent = uniform01(state)
        selector = uevent * total_rate
        state, node = random_active(state, active_nodes, active_count)
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
            out[event_bin, 0] += 1.0
        elif selector < infection_rate + recovery_rate:
            active_count, m1 = remove_node(
                node, active, parent, active_child_count, active_nodes, active_pos, active_count, m1
            )
            out[event_bin, 1] += 1.0
        else:
            nt = 0
            active_pf = pf if next_tau >= trace_on else 0.0
            if active_pf > 0.0:
                child = first_child[node]
                while child >= 0:
                    if active[child] == 1:
                        take = False
                        if active_pf >= 1.0:
                            take = True
                        else:
                            state, ur = uniform01(state)
                            if ur < active_pf:
                                take = True
                        if take:
                            traced_buf[nt] = child
                            nt += 1
                    child = next_sibling[child]
            active_count, m1 = remove_node(
                node, active, parent, active_child_count, active_nodes, active_pos, active_count, m1
            )
            for q in range(nt):
                child = traced_buf[q]
                if active[child] == 1:
                    active_count, m1 = remove_node(
                        child, active, parent, active_child_count, active_nodes, active_pos, active_count, m1
                    )
            out[event_bin, 2] += 1.0
            out[event_bin, 3] += nt
        if m1 < 0 or m1 > active_count:
            raise AssertionError("invalid M1 bookkeeping")
        time = next_time
        tau = next_tau
    return out


@njit(parallel=True)
def simulate_batch(U0_, I0_, beta, gamma, gamma_c, pf, Gamma, tau_end, n_bins, trace_on, seeds):
    result = np.zeros((seeds.size, n_bins, 7), np.float64)
    for r in prange(seeds.size):
        result[r] = simulate_one_numba(
            U0_, I0_, beta, gamma, gamma_c, pf, Gamma, tau_end, n_bins, trace_on, seeds[r]
        )
    return result


def analytical_expected(protocol: Protocol, exposure_mean: np.ndarray) -> np.ndarray:
    expected = np.zeros((N_BINS, 4), dtype=float)
    expected[:, 0] = protocol.beta / protocol.Gamma * exposure_mean[:, 0]
    expected[:, 1] = protocol.gamma / protocol.Gamma * exposure_mean[:, 1]
    expected[:, 2] = protocol.gamma_c / protocol.Gamma * exposure_mean[:, 1]
    expected[:, 3] = protocol.gamma_c * protocol.p_f / protocol.Gamma * exposure_mean[:, 2]
    return expected


def set_bmb_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.0,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    slope = float(np.dot(x, y) / np.dot(x, x))
    residual = y - slope*x
    sst = float(np.sum((y - np.mean(y))**2))
    return {
        "n_points": int(len(x)),
        "slope_through_origin": slope,
        "r_squared": float(1 - np.sum(residual**2)/sst),
        "total_observed_over_expected": float(np.sum(y)/np.sum(x)),
        "rmse_events_per_realization_interval": float(np.sqrt(np.mean((y-x)**2))),
        "mae_events_per_realization_interval": float(np.mean(np.abs(y-x))),
    }


def compute_panel_summaries(rows) -> list[dict]:
    """Per-event calibration statistics, independent of any plotting."""
    summaries = []
    for event in EVENT_NAMES:
        x = np.array([r[f"expected_{event}"] for r in rows], float)
        y = np.array([r[f"observed_{event}"] for r in rows], float)
        if event == "traced_removal":
            mask = (x > 0) | (y > 0)
            omitted = int(np.sum(~mask))
            xp, yp = x[mask], y[mask]
        else:
            omitted = 0
            xp, yp = x, y
        stat = metrics(xp, yp)
        stat.update({"event": event, "zero_hazard_points_omitted_from_plot": omitted})
        summaries.append(stat)
    return summaries


def plot_panels(rows, summaries, out_dir):
    """Render the four scatter panels from precomputed rows + summaries."""
    set_bmb_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True)
    for i, (ax, event, title, stat) in enumerate(zip(axes.flat, EVENT_NAMES, EVENT_LABELS, summaries)):
        x = np.array([r[f"expected_{event}"] for r in rows], float)
        y = np.array([r[f"observed_{event}"] for r in rows], float)
        if event == "traced_removal":
            mask = (x > 0) | (y > 0)
            xp, yp = x[mask], y[mask]
        else:
            xp, yp = x, y
        lim = max(float(np.max(xp)), float(np.max(yp))) * 1.035
        ax.scatter(xp, yp, s=5.0, alpha=0.18, linewidths=0, rasterized=True)
        ax.plot([0, lim], [0, lim], color="black", lw=1.0, ls=(0,(4,2)))
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"({chr(97+i)}) {title}", loc="left", pad=5)
        ax.set_xlabel("State-supplied expected events\nper realization and interval")
        ax.set_ylabel("Observed ABM events\nper realization and interval")
        ax.text(0.04, 0.95, rf"slope $={stat['slope_through_origin']:.4f}$"+"\n"+rf"$R^2={stat['r_squared']:.4f}$",
                transform=ax.transAxes, va="top", ha="left",
                bbox={"boxstyle":"round,pad=0.25","facecolor":"white","alpha":0.86,"edgecolor":"0.75"})
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3,4), useMathText=True)
    fig.savefig(out_dir/"supp_S5_event_term_validation.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir/"supp_S5_event_term_validation.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image
    with Image.open(out_dir/"supp_S5_event_term_validation.png") as im:
        im.convert("RGB").save(out_dir/"supp_S5_event_term_validation.tiff", compression="tiff_lzw", dpi=(600,600))
