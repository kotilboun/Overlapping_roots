#!/usr/bin/env python3
"""ABM ensemble-mean trajectory error over population and replicate scales.

The script uses the Section 7 nondimensional grid
    R in {2,4,6}, c in {0,0.25,0.5,0.75,1}
and, for each cell, simulates multiple dimensional raw-rate decompositions
(beta, gamma, gamma_c, p_f) that share the same nondimensional model.

For each raw protocol and population scale, a master pool of ABM trajectories
is generated in physical time and transformed afterward to tau=Gamma*t.
Subsets of the master pool are used to quantify the error between the ABM
replicate-mean trajectory and a converged high-depth deterministic hierarchy as
a function of population scale and number of replicates in the mean.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "abm-mean-error-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp


R_GRID = (2.0, 4.0, 6.0)
C_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
U0_GRID = (500, 1000, 2000, 5000)
I0_FRACTION = 0.02
MASTER_REPLICATES = 20
REPLICATE_SIZES = (1, 2, 5, 10, 20)
SUBSET_DRAWS = 100
TAU_END = 5.0
NUM_TIMES = 151
SWITCH_TAU = 0.5
MASTER_SEED = 20260729
HIGH_DEPTH_ORDER = 40
CHECK_DEPTH_ORDER = 50


@dataclass(frozen=True)
class RawProtocol:
    protocol_id: str
    Gamma: float
    d: float
    beta: float
    gamma: float
    gamma_c: float
    p_f: float
    R: float
    c: float
    U0: int
    I0: int


class ActiveGenealogy:
    """Active transmission forest with O(1) M1 updates."""

    def __init__(self, initial_infectious: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.parent: list[int] = []
        self.children: list[list[int]] = []
        self.active: list[bool] = []
        self.active_nodes: list[int] = []
        self.active_position: list[int] = []
        self.active_child_count: list[int] = []
        self.m1 = 0
        for _ in range(initial_infectious):
            self.add_root()

    def add_root(self) -> int:
        node = len(self.parent)
        self.parent.append(-1)
        self.children.append([])
        self.active.append(True)
        self.active_position.append(len(self.active_nodes))
        self.active_child_count.append(0)
        self.active_nodes.append(node)
        return node

    def random_active(self) -> int:
        return self.active_nodes[int(self.rng.integers(len(self.active_nodes)))]

    def add_child(self, parent: int) -> int:
        node = self.add_root()
        self.parent[node] = parent
        self.children[parent].append(node)
        self.active_child_count[parent] += 1
        self.m1 += 1
        return node

    def remove(self, node: int) -> None:
        if not self.active[node]:
            return

        parent = self.parent[node]
        if parent >= 0 and self.active[parent]:
            self.active_child_count[parent] -= 1
            self.m1 -= 1

        # Removing the node destroys all active parent-child relations rooted at it.
        self.m1 -= self.active_child_count[node]
        if self.m1 < 0:
            raise AssertionError("M1 became negative")

        self.active[node] = False
        position = self.active_position[node]
        last = self.active_nodes.pop()
        if last != node:
            self.active_nodes[position] = last
            self.active_position[last] = position
        self.active_position[node] = -1

    def detect_and_trace(self, node: int, p_f: float) -> int:
        if p_f <= 0.0:
            self.remove(node)
            return 0

        active_children = [child for child in self.children[node] if self.active[child]]
        if p_f >= 1.0:
            traced = active_children
        else:
            traced = [child for child in active_children if self.rng.random() < p_f]

        self.remove(node)
        for child in traced:
            self.remove(child)
        return len(traced)

    def direct_m1(self) -> int:
        return sum(
            1
            for node in self.active_nodes
            for child in self.children[node]
            if self.active[child]
        )


def validate_bookkeeping(seed: int) -> None:
    rng = np.random.default_rng(seed)
    tree = ActiveGenealogy(12, rng)
    for _ in range(4000):
        if not tree.active_nodes:
            tree.add_root()
        draw = rng.random()
        if draw < 0.55:
            tree.add_child(tree.random_active())
        elif draw < 0.78:
            tree.remove(tree.random_active())
        else:
            tree.detect_and_trace(tree.random_active(), float(rng.random()))
        if tree.m1 != tree.direct_m1():
            raise AssertionError(
                f"Incremental and direct M1 disagree: {tree.m1} != {tree.direct_m1()}"
            )


def d_values_for_c(c: float) -> tuple[float, ...]:
    if c == 0.0:
        return (0.2, 0.4, 0.6, 0.8, 1.0)
    if c == 1.0:
        return (1.0,)
    return tuple(float(x) for x in np.linspace(c, 1.0, 5))


def build_protocols(R: float, c: float, U0: int, I0: int) -> list[RawProtocol]:
    """Four raw-rate protocols spanning time-scale and decomposition extremes."""
    protocols: list[RawProtocol] = []

    if c == 1.0:
        # At c=1, d=p_f=1 is forced; only the dimensional time scale can vary.
        pairs = [(Gamma, 1.0) for Gamma in (0.5, 1.0, 2.0, 4.0)]
    else:
        low_d = 0.25 if c == 0.0 else c
        # Extremes: less frequent complete tracing (d=c,p_f=1) and
        # frequent partial tracing (d=1,p_f=c), at two dimensional time scales.
        pairs = [(0.5, low_d), (0.5, 1.0), (2.0, low_d), (2.0, 1.0)]

    for index, (Gamma, d) in enumerate(pairs, start=1):
        p_f = 0.0 if c == 0.0 else c / d
        gamma_c = d * Gamma
        gamma = (1.0 - d) * Gamma
        beta = R * Gamma / U0
        protocols.append(
            RawProtocol(
                protocol_id=f"P{index:02d}", Gamma=Gamma, d=d,
                beta=beta, gamma=gamma, gamma_c=gamma_c, p_f=p_f,
                R=R, c=c, U0=U0, I0=I0,
            )
        )
    return protocols


def simulate_one(
    protocol: RawProtocol,
    tau_grid: np.ndarray,
    switch_tau: float,
    seed: int,
) -> np.ndarray:
    """Return U, I, M1 counts along one ABM trajectory."""
    rng = np.random.default_rng(seed)
    tree = ActiveGenealogy(protocol.I0, rng)
    U = protocol.U0
    output = np.zeros((len(tau_grid), 3), dtype=np.float32)
    physical_times = tau_grid / protocol.Gamma
    switch_time = switch_tau / protocol.Gamma
    index = 0
    time = 0.0

    def record_until(limit: float) -> None:
        nonlocal index
        while index < len(physical_times) and physical_times[index] <= limit:
            output[index, 0] = U
            output[index, 1] = len(tree.active_nodes)
            output[index, 2] = tree.m1
            index += 1

    record_until(0.0)

    while index < len(physical_times):
        I = len(tree.active_nodes)
        if I == 0:
            output[index:, 0] = U
            output[index:, 1:] = 0.0
            break

        infection_rate = protocol.beta * U * I if U > 0 else 0.0
        spontaneous_rate = protocol.gamma * I
        detection_rate = protocol.gamma_c * I
        total_rate = infection_rate + spontaneous_rate + detection_rate

        if total_rate <= 0.0:
            output[index:, 0] = U
            output[index:, 1] = I
            output[index:, 2] = tree.m1
            break

        next_time = time + float(rng.exponential(1.0 / total_rate))
        record_until(next_time)
        if index >= len(physical_times):
            break
        time = next_time

        selector = rng.random() * total_rate
        if selector < infection_rate:
            tree.add_child(tree.random_active())
            U -= 1
        elif selector < infection_rate + spontaneous_rate:
            tree.remove(tree.random_active())
        else:
            active_pf = protocol.p_f if time >= switch_time else 0.0
            tree.detect_and_trace(tree.random_active(), active_pf)

    return output


def simulate_task(args: tuple[RawProtocol, np.ndarray, float, int, int]) -> tuple[RawProtocol, np.ndarray]:
    protocol, tau_grid, switch_tau, master_replicates, seed_base = args
    trajectories = np.zeros((master_replicates, len(tau_grid), 3), dtype=np.float32)
    for replicate in range(master_replicates):
        seed_sequence = np.random.SeedSequence(
            [seed_base, protocol.U0, int(protocol.R * 100), int(protocol.c * 1000),
             int(protocol.Gamma * 10000), int(protocol.d * 10000), replicate]
        )
        seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
        trajectories[replicate] = simulate_one(protocol, tau_grid, switch_tau, seed)
    return protocol, trajectories


def solve_high_depth(
    R: float,
    c: float,
    tau_grid: np.ndarray,
    switch_tau: float,
    order: int,
    initial_infectious_fraction: float,
) -> np.ndarray:
    """Solve scaled count hierarchy: z=(U/U0,I/U0,M1/U0,...)."""
    state0 = np.zeros(order + 2, dtype=float)
    state0[0] = 1.0
    state0[1] = initial_infectious_fraction

    def rhs(active_c: float):
        def evaluate(_tau: float, state: np.ndarray) -> np.ndarray:
            U = max(float(state[0]), 0.0)
            I = max(float(state[1]), 0.0)
            moments = state[2:]
            R_U = R * U
            derivative = np.zeros_like(state)
            derivative[0] = -R_U * I
            derivative[1] = (R_U - 1.0) * I - active_c * moments[0]
            for k in range(1, order + 1):
                previous = I if k == 1 else moments[k - 2]
                current = moments[k - 1]
                following = moments[k] if k < order else 0.0
                derivative[k + 1] = (
                    R_U * previous - (k + 1) * current - active_c * following
                )
            return derivative
        return evaluate

    pre_mask = tau_grid <= switch_tau
    pre_times = tau_grid[pre_mask]
    pre_eval = np.unique(np.append(pre_times, switch_tau))
    first = solve_ivp(
        rhs(0.0), (float(tau_grid[0]), switch_tau), state0,
        t_eval=pre_eval, method="BDF", rtol=1e-10, atol=1e-12, max_step=0.025,
    )
    if not first.success:
        raise RuntimeError(first.message)

    output = np.empty((len(tau_grid), len(state0)), dtype=float)
    output[pre_mask] = first.y[:, : len(pre_times)].T

    post_mask = tau_grid > switch_tau
    if np.any(post_mask):
        second = solve_ivp(
            rhs(c), (switch_tau, float(tau_grid[-1])), first.y[:, -1],
            t_eval=tau_grid[post_mask], method="BDF", rtol=1e-10, atol=1e-12,
            max_step=0.025,
        )
        if not second.success:
            raise RuntimeError(second.message)
        output[post_mask] = second.y.T
    return output


def integrated_rmse(values: np.ndarray, reference: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Time-normalized integrated RMSE for one or more trajectories."""
    difference_sq = np.square(values - reference)
    integral = np.trapezoid(difference_sq, tau, axis=-2) / (tau[-1] - tau[0])
    return np.sqrt(integral)


def subset_error_samples(
    normalized_raw: np.ndarray,
    deterministic: np.ndarray,
    tau: np.ndarray,
    replicate_sizes: Iterable[int],
    draws: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Errors for replicate-mean trajectories from random subsets."""
    rng = np.random.default_rng(seed)
    master_replicates = normalized_raw.shape[0]
    rows: list[dict[str, float | int]] = []

    for number in replicate_sizes:
        if number > master_replicates:
            continue
        draw_count = 1 if number == master_replicates else draws
        for subset_id in range(draw_count):
            if number == master_replicates:
                indices = np.arange(master_replicates)
            else:
                indices = rng.choice(master_replicates, size=number, replace=False)
            mean_trajectory = normalized_raw[indices].mean(axis=0)
            component_errors = integrated_rmse(
                mean_trajectory[None, :, :], deterministic[:, :3], tau
            )[0]
            combined = float(np.linalg.norm(component_errors))
            rows.append(
                {
                    "n_replicates": int(number),
                    "subset_id": int(subset_id),
                    "E_U": float(component_errors[0]),
                    "E_I": float(component_errors[1]),
                    "E_M1": float(component_errors[2]),
                    "E_trajectory": combined,
                }
            )
    return rows


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path_stem: Path) -> tuple[Path, Path, Path]:
    pdf = path_stem.with_suffix(".pdf")
    png = path_stem.with_suffix(".png")
    tiff = path_stem.with_suffix(".tiff")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    for path, compression in ((png, None), (tiff, "tiff_lzw")):
        with Image.open(path) as image:
            kwargs: dict[str, object] = {"dpi": (600, 600)}
            if compression:
                kwargs["compression"] = compression
            image.convert("RGB").save(path, **kwargs)
    return pdf, png, tiff


def quantile_summary(error_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int]]:
    groups: dict[tuple[int, int], list[float]] = {}
    for row in error_rows:
        key = (int(row["U0"]), int(row["n_replicates"]))
        groups.setdefault(key, []).append(float(row["E_trajectory"]))

    result: list[dict[str, float | int]] = []
    for (U0, nrep), values in sorted(groups.items()):
        array = np.asarray(values, dtype=float)
        result.append(
            {
                "U0": U0,
                "n_replicates": nrep,
                "n_error_samples": len(array),
                "median_E_trajectory": float(np.quantile(array, 0.50)),
                "q25_E_trajectory": float(np.quantile(array, 0.25)),
                "q75_E_trajectory": float(np.quantile(array, 0.75)),
                "q90_E_trajectory": float(np.quantile(array, 0.90)),
                "q95_E_trajectory": float(np.quantile(array, 0.95)),
            }
        )
    return result


def make_figures(summary: list[dict[str, float | int]], output_dir: Path) -> list[Path]:
    set_publication_style()
    U_values = sorted({int(row["U0"]) for row in summary})
    replicate_values = sorted({int(row["n_replicates"]) for row in summary})
    lookup = {(int(row["U0"]), int(row["n_replicates"])): row for row in summary}

    median_matrix = np.asarray(
        [[float(lookup[(U0, n)]["median_E_trajectory"]) for U0 in U_values]
         for n in replicate_values]
    )
    q90_matrix = np.asarray(
        [[float(lookup[(U0, n)]["q90_E_trajectory"]) for U0 in U_values]
         for n in replicate_values]
    )

    created: list[Path] = []

    # Figure 1: median heatmap.
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    image = ax.imshow(median_matrix, aspect="auto", origin="lower")
    ax.set_xticks(range(len(U_values)), [f"{value:,}" for value in U_values])
    ax.set_yticks(range(len(replicate_values)), [str(value) for value in replicate_values])
    ax.set_xlabel(r"population scale $S_0$")
    ax.set_ylabel("replicates in ABM mean")
    ax.set_title(r"Median ensemble-mean trajectory error $E_{\mathrm{traj}}$")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"median $E_{\mathrm{traj}}$")
    for row_index in range(len(replicate_values)):
        for column_index in range(len(U_values)):
            ax.text(column_index, row_index, f"{median_matrix[row_index, column_index]:.3g}",
                    ha="center", va="center", fontsize=8)
    created.extend(save_figure(fig, output_dir / "fig_median_trajectory_error_heatmap"))

    # Figure 2: upper-tail heatmap.
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    image = ax.imshow(q90_matrix, aspect="auto", origin="lower")
    ax.set_xticks(range(len(U_values)), [f"{value:,}" for value in U_values])
    ax.set_yticks(range(len(replicate_values)), [str(value) for value in replicate_values])
    ax.set_xlabel(r"population scale $S_0$")
    ax.set_ylabel("replicates in ABM mean")
    ax.set_title(r"90th-percentile ensemble-mean trajectory error")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"90th percentile of $E_{\mathrm{traj}}$")
    for row_index in range(len(replicate_values)):
        for column_index in range(len(U_values)):
            ax.text(column_index, row_index, f"{q90_matrix[row_index, column_index]:.3g}",
                    ha="center", va="center", fontsize=8)
    created.extend(save_figure(fig, output_dir / "fig_q90_trajectory_error_heatmap"))

    # Figure 3: error versus population scale.
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for nrep in replicate_values:
        medians = [float(lookup[(U0, nrep)]["median_E_trajectory"]) for U0 in U_values]
        lower = [float(lookup[(U0, nrep)]["q25_E_trajectory"]) for U0 in U_values]
        upper = [float(lookup[(U0, nrep)]["q75_E_trajectory"]) for U0 in U_values]
        line, = ax.plot(U_values, medians, marker="o", label=f"{nrep} replicates")
        ax.fill_between(U_values, lower, upper, alpha=0.12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"population scale $S_0$")
    ax.set_ylabel(r"trajectory error $E_{\mathrm{traj}}$")
    ax.set_title("ABM ensemble-mean error versus population scale")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    created.extend(save_figure(fig, output_dir / "fig_trajectory_error_vs_population"))

    # Figure 4: error versus replicate count.
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for U0 in U_values:
        medians = [float(lookup[(U0, nrep)]["median_E_trajectory"]) for nrep in replicate_values]
        lower = [float(lookup[(U0, nrep)]["q25_E_trajectory"]) for nrep in replicate_values]
        upper = [float(lookup[(U0, nrep)]["q75_E_trajectory"]) for nrep in replicate_values]
        ax.plot(replicate_values, medians, marker="o", label=rf"$S_0={U0:,}$")
        ax.fill_between(replicate_values, lower, upper, alpha=0.12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("replicates in ABM mean")
    ax.set_ylabel(r"trajectory error $E_{\mathrm{traj}}$")
    ax.set_title("ABM ensemble-mean error versus replicate count")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    created.extend(save_figure(fig, output_dir / "fig_trajectory_error_vs_replicates"))

    return created


def write_caption_and_methods(output_dir: Path) -> None:
    methods = r"""## Supplementary analysis: convergence of the ABM ensemble-mean trajectory

For each nondimensional parameter cell on the Section 7 grid,
\[
R\in\{2,4,6\},\qquad c\in\{0,0.25,0.5,0.75,1\},
\]
we generated multiple dimensional parameterizations \((\beta,\gamma,\gamma_c,p_f)\)
that preserved
\[
R=\frac{\beta U_0}{\Gamma},\qquad
c=\frac{\gamma_c p_f}{\Gamma},\qquad
\Gamma=\gamma+\gamma_c.
\]
The ABM was simulated in physical time using the raw rates \(\beta UI\),
\(\gamma I\), and \(\gamma_c I\), and the resulting trajectories were transformed
afterward using \(\tau=\Gamma t\). Initial conditions satisfied
\(I_0/U_0=0.02\), with unrelated initial infectious roots, and tracing was
activated at \(\tau=0.5\).

Let
\[
\mathbf z(\tau)=\left(\frac{U(\tau)}{U_0},
\frac{I(\tau)}{U_0},\frac{M_1(\tau)}{U_0}\right)
\]
denote the scaled deterministic state and let
\(\overline{\mathbf z}^{\mathrm{ABM}}_{n}(\tau)\) denote the mean of \(n\)
independent ABM trajectories. We define the ensemble-mean trajectory error by
\[
E_{\mathrm{traj}}=
\left[
\frac{1}{T}\int_0^T
\left\|\overline{\mathbf z}^{\mathrm{ABM}}_{n}(\tau)
-\mathbf z(\tau)\right\|_2^2\,\mathrm d\tau
\right]^{1/2}.
\]
The deterministic reference was obtained from a high-depth hierarchy, and its
numerical convergence was checked against a deeper truncation. For each raw
protocol and population scale, a master pool of independent ABM realizations was
generated. Random subsets of sizes \(n\in\{1,2,5,10,20\}\) were used to form
ABM replicate means, and the resulting error distribution was summarized across
the full \((R,c)\) grid, raw-rate parameterizations, and subset resamples.
"""
    (output_dir / "supplementary_methods_trajectory_error.md").write_text(methods, encoding="utf-8")

    captions = r"""**Supplementary Figure Sx. Population- and replicate-scale convergence of the ABM ensemble mean.** The trajectory error compares the ABM replicate-mean trajectory with the common high-depth deterministic hierarchy using the scaled state \((U/U_0,I/U_0,M_1/U_0)\). The analysis covers \(R\in\{2,4,6\}\), \(c\in\{0,0.25,0.5,0.75,1\}\), five population scales, and multiple dimensional decompositions \((\beta,\gamma,\gamma_c,p_f)\) preserving the same nondimensional parameters. Simulations were performed in physical time and transformed afterward using \(\tau=\Gamma t\). Heatmaps report the median and 90th percentile of the integrated trajectory error across parameter cells, raw protocols, and replicate-subset resamples. The line figures show the median error with interquartile bands as a function of population scale and the number of trajectories used in the ABM mean.
"""
    (output_dir / "supplementary_figure_caption.md").write_text(captions, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--master-replicates", type=int, default=MASTER_REPLICATES)
    parser.add_argument("--subset-draws", type=int, default=SUBSET_DRAWS)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tau = np.linspace(0.0, TAU_END, NUM_TIMES)
    validate_bookkeeping(MASTER_SEED + 17)

    # Deterministic references and depth convergence.
    deterministic: dict[tuple[float, float], np.ndarray] = {}
    depth_checks: list[dict[str, float]] = []
    for R in R_GRID:
        for c in C_GRID:
            accepted = solve_high_depth(R, c, tau, SWITCH_TAU, HIGH_DEPTH_ORDER, I0_FRACTION)
            deeper = solve_high_depth(R, c, tau, SWITCH_TAU, CHECK_DEPTH_ORDER, I0_FRACTION)
            deterministic[(R, c)] = accepted
            maximum_difference = float(np.max(np.abs(accepted[:, :3] - deeper[:, :3])))
            depth_checks.append({"R": R, "c": c, "max_order40_vs_order50_difference": maximum_difference})

    with (output_dir / "high_depth_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(depth_checks[0].keys()))
        writer.writeheader()
        writer.writerows(depth_checks)

    # Generate all master pools.
    task_arguments = []
    protocols_all: list[RawProtocol] = []
    for U0 in U0_GRID:
        I0 = int(round(I0_FRACTION * U0))
        for R in R_GRID:
            for c in C_GRID:
                protocols = build_protocols(R, c, U0, I0)
                protocols_all.extend(protocols)
                for protocol in protocols:
                    task_arguments.append((protocol, tau, SWITCH_TAU, args.master_replicates, MASTER_SEED))

    master_data: dict[tuple[int, float, float, str], np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(simulate_task, task) for task in task_arguments]
        for completed_index, future in enumerate(as_completed(futures), start=1):
            protocol, trajectories = future.result()
            master_data[(protocol.U0, protocol.R, protocol.c, protocol.protocol_id)] = trajectories
            if completed_index % 50 == 0:
                print(f"Completed {completed_index}/{len(futures)} raw protocols", flush=True)

    protocol_rows = [asdict(protocol) for protocol in protocols_all]
    with (output_dir / "raw_protocol_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(protocol_rows[0].keys()))
        writer.writeheader()
        writer.writerows(protocol_rows)

    # Save master pools by population scale.
    for U0 in U0_GRID:
        payload: dict[str, np.ndarray] = {"tau": tau}
        for protocol in (p for p in protocols_all if p.U0 == U0):
            key = f"R{protocol.R:g}_c{protocol.c:g}_{protocol.protocol_id}".replace(".", "p")
            payload[key] = master_data[(protocol.U0, protocol.R, protocol.c, protocol.protocol_id)]
        np.savez_compressed(output_dir / f"master_abm_trajectories_U0_{U0}.npz", **payload)

    # Subset error analysis.
    error_rows: list[dict[str, float | int | str]] = []
    for protocol_index, protocol in enumerate(protocols_all):
        raw = master_data[(protocol.U0, protocol.R, protocol.c, protocol.protocol_id)].astype(float)
        normalized = raw / float(protocol.U0)
        reference = deterministic[(protocol.R, protocol.c)][:, :3]
        subset_rows = subset_error_samples(
            normalized,
            reference,
            tau,
            REPLICATE_SIZES,
            args.subset_draws,
            seed=MASTER_SEED + 100003 * protocol_index,
        )
        for row in subset_rows:
            error_rows.append(
                {
                    "U0": protocol.U0,
                    "I0": protocol.I0,
                    "R": protocol.R,
                    "c": protocol.c,
                    "protocol_id": protocol.protocol_id,
                    "Gamma": protocol.Gamma,
                    "d": protocol.d,
                    "beta": protocol.beta,
                    "gamma": protocol.gamma,
                    "gamma_c": protocol.gamma_c,
                    "p_f": protocol.p_f,
                    **row,
                }
            )

    error_fieldnames = list(error_rows[0].keys())
    with gzip.open(output_dir / "trajectory_error_samples.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)

    summary = quantile_summary(error_rows)
    with (output_dir / "trajectory_error_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    make_figures(summary, output_dir)
    write_caption_and_methods(output_dir)

    parameters = {
        "R_grid": list(R_GRID),
        "c_grid": list(C_GRID),
        "U0_grid": list(U0_GRID),
        "I0_fraction": I0_FRACTION,
        "master_replicates": args.master_replicates,
        "replicate_sizes": list(REPLICATE_SIZES),
        "subset_draws": args.subset_draws,
        "tau_end": TAU_END,
        "num_times": NUM_TIMES,
        "switch_tau": SWITCH_TAU,
        "master_seed": MASTER_SEED,
        "high_depth_order": HIGH_DEPTH_ORDER,
        "check_depth_order": CHECK_DEPTH_ORDER,
        "trajectory_state": ["U/U0", "I/U0", "M1/U0"],
        "trajectory_error": "sqrt((1/T) integral ||mean_ABM-z_det||_2^2 dtau)",
    }
    (output_dir / "analysis_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    print("Analysis complete", flush=True)


if __name__ == "__main__":
    main()
