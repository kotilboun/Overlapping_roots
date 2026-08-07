"""Canonical selected-QSS continuation solver used by v2 analytical figures.

The nonlinear solution is selected by continuation from the exact c=0 state.
For each trial root variable ``a = R - c m1``, the minimal linear tail is
evaluated by the zero-terminal backward continued fraction

    y_k = c R / (a + k + y_{k+1}).

Depth is doubled until the nonlinear root and the continued-fraction/Bessel
cross-check satisfy the declared tolerances.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log, sqrt
from typing import Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.special import ive


@dataclass(frozen=True)
class SolverConfig:
    initial_depth: int = 20
    max_depth: int = 640
    depth_absolute_tolerance: float = 1.0e-12
    depth_relative_tolerance: float = 1.0e-11
    fixed_point_residual_tolerance: float = 1.0e-10
    bessel_crosscheck_tolerance: float = 2.0e-10
    root_xtol: float = 1.0e-14
    maximum_reported_moment_depth: int = 10

    def validate(self) -> None:
        if self.initial_depth < 4:
            raise ValueError("initial_depth must be at least four.")
        if self.max_depth < 2 * self.initial_depth:
            raise ValueError("max_depth must permit at least one depth doubling.")
        if self.max_depth % self.initial_depth:
            raise ValueError("max_depth must be a multiple of initial_depth.")
        if self.maximum_reported_moment_depth >= self.initial_depth:
            raise ValueError("Reported moment depth must be below initial depth.")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class QSSResult:
    R: float
    c: float
    m1: float
    a: float
    moments: tuple[float, ...]
    ratios: tuple[float, ...]
    accepted_depth: int
    previous_depth: int
    depth_doubling_difference: float
    fixed_point_residual: float
    continued_fraction_tail_bound_y1: float
    a_posteriori_root_error_estimate: float
    bessel_m1: float
    bessel_cf_difference: float
    simple_root_margin: float
    minimum_moment: float
    minimum_monotonicity_margin: float
    admissible: bool
    continuation_predecessor_a: float
    continuation_predictor_a: float
    root_bracket_left: float
    root_bracket_right: float
    bracket_expansions: int
    bracket_source: str

    def to_row(self) -> dict[str, float | int | str | bool]:
        row: dict[str, float | int | str | bool] = {
            "R": self.R,
            "c": self.c,
            "m1": self.m1,
            "a": self.a,
            "accepted_depth": self.accepted_depth,
            "previous_depth": self.previous_depth,
            "depth_doubling_difference": self.depth_doubling_difference,
            "fixed_point_residual": self.fixed_point_residual,
            "continued_fraction_tail_bound_y1": self.continued_fraction_tail_bound_y1,
            "a_posteriori_root_error_estimate": self.a_posteriori_root_error_estimate,
            "bessel_m1": self.bessel_m1,
            "bessel_cf_difference": self.bessel_cf_difference,
            "simple_root_margin": self.simple_root_margin,
            "minimum_moment": self.minimum_moment,
            "minimum_monotonicity_margin": self.minimum_monotonicity_margin,
            "admissible": self.admissible,
            "continuation_predecessor_a": self.continuation_predecessor_a,
            "continuation_predictor_a": self.continuation_predictor_a,
            "root_bracket_left": self.root_bracket_left,
            "root_bracket_right": self.root_bracket_right,
            "bracket_expansions": self.bracket_expansions,
            "bracket_source": self.bracket_source,
        }
        for depth, value in enumerate(self.moments):
            row[f"m{depth}"] = value
        for depth, value in enumerate(self.ratios, start=1):
            row[f"r{depth}"] = value
        return row


@dataclass(frozen=True)
class FixedDepthResult:
    R: float
    c: float
    depth: int
    m1: float
    a: float
    fixed_point_residual: float
    bessel_cf_difference: float
    simple_root_margin: float
    admissible: bool
    bracket_left: float
    bracket_right: float
    bracket_expansions: int
    bracket_source: str


def continued_fraction_tail(
    a: float,
    x: float,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return y_1,...,y_depth and their derivatives with respect to a."""
    if a <= -1.0:
        raise ValueError(f"Continued fraction requires a>-1; received {a}.")
    if x < 0.0:
        raise ValueError("Continued fraction requires x>=0.")
    y = np.zeros(depth + 2, dtype=float)
    derivative = np.zeros(depth + 2, dtype=float)
    for k in range(depth, 0, -1):
        denominator = a + k + y[k + 1]
        if denominator <= 0.0 or not isfinite(denominator):
            raise FloatingPointError(
                f"Nonpositive/nonfinite continued-fraction denominator at k={k}: "
                f"a={a}, x={x}, denominator={denominator}."
            )
        y[k] = x / denominator
        derivative[k] = -x * (1.0 + derivative[k + 1]) / denominator**2
    return y, derivative


def stable_bessel_y1(a: float, x: float) -> float:
    """Evaluate sqrt(x) I_{a+1}(2 sqrt(x))/I_a(2 sqrt(x)) stably."""
    if x == 0.0:
        return 0.0
    z = 2.0 * sqrt(x)
    denominator = float(ive(a, z))
    numerator = float(ive(a + 1.0, z))
    if (
        not isfinite(denominator)
        or not isfinite(numerator)
        or denominator == 0.0
    ):
        raise FloatingPointError(
            f"Scaled Bessel ratio failed for a={a}, x={x}: "
            f"Ive(a)={denominator}, Ive(a+1)={numerator}."
        )
    value = sqrt(x) * numerator / denominator
    if value <= 0.0 or not isfinite(value):
        raise FloatingPointError(
            f"Scaled Bessel ratio was not positive and finite: {value}."
        )
    return value


def continued_fraction_y1_bound(a: float, x: float, depth: int) -> float:
    """Analytical zero-terminal error bound for y_1 at fixed a."""
    if x == 0.0:
        return 0.0
    log_bound = (
        (depth + 1) * log(x)
        - log(a + depth + 1.0)
        - 2.0 * sum(log(a + j) for j in range(1, depth + 1))
    )
    if log_bound < log(np.finfo(float).tiny):
        return 0.0
    return exp(log_bound)


def _root_at_depth(
    R: float,
    c: float,
    depth: int,
    continuation_center_a: float,
    root_xtol: float = 1.0e-13,
) -> tuple[float, tuple[float, float], int, str]:
    lower = R - c
    upper = R
    x = c * R

    def equation(a: float) -> float:
        y, _ = continued_fraction_tail(a, x, depth)
        return a + y[1] - R

    center = min(max(continuation_center_a, lower), upper)
    maximum_width = upper - lower
    width = min(max(5.0e-4, 0.02 * max(1.0, maximum_width)), maximum_width)
    expansions = 0
    bracket_source = "local_continuation"
    left = max(lower, center - width)
    right = min(upper, center + width)

    while True:
        f_left = equation(left)
        f_right = equation(right)
        if f_left == 0.0:
            return left, (left, left), expansions, bracket_source
        if f_right == 0.0:
            return right, (right, right), expansions, bracket_source
        if f_left * f_right < 0.0:
            root = float(
                brentq(
                    equation,
                    left,
                    right,
                    xtol=root_xtol,
                    rtol=4.0 * np.finfo(float).eps,
                    maxiter=200,
                )
            )
            return root, (left, right), expansions, bracket_source
        if left <= lower and right >= upper:
            break
        width = min(maximum_width, width * 2.0)
        left = max(lower, center - width)
        right = min(upper, center + width)
        expansions += 1

    # The full existence bracket is a fail-safe, not the selection rule.
    bracket_source = "full_existence_fallback"
    left, right = lower, upper
    f_left, f_right = equation(left), equation(right)
    if not (f_left < 0.0 < f_right):
        raise RuntimeError(
            "Finite continued fraction did not preserve the full endpoint signs: "
            f"R={R}, c={c}, depth={depth}, G(lower)={f_left}, G(upper)={f_right}."
        )
    root = float(
        brentq(
            equation,
            left,
            right,
            xtol=root_xtol,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=200,
        )
    )
    return root, (left, right), expansions, bracket_source


def _moments_and_diagnostics(
    R: float,
    c: float,
    a: float,
    depth: int,
    maximum_moment_depth: int,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    float,
    float,
    float,
    bool,
]:
    if c == 0.0:
        ratios = tuple(R / (R + k) for k in range(1, maximum_moment_depth + 1))
    else:
        y, derivative = continued_fraction_tail(a, c * R, depth)
        ratios = tuple(float(y[k] / c) for k in range(1, maximum_moment_depth + 1))
    moments = [1.0]
    for ratio in ratios:
        moments.append(moments[-1] * ratio)
    minimum_moment = min(moments)
    margins = [moments[k - 1] - moments[k] for k in range(1, len(moments))]
    minimum_margin = min(margins)
    admissible = (
        all(isfinite(value) and value > 0.0 for value in moments)
        and all(0.0 < ratio < 1.0 for ratio in ratios)
        and minimum_margin > 0.0
    )
    if c == 0.0:
        simple_margin = 1.0
    else:
        _, derivative = continued_fraction_tail(a, c * R, depth)
        simple_margin = abs(1.0 + float(derivative[1]))
    return (
        tuple(float(value) for value in moments),
        ratios,
        float(simple_margin),
        float(minimum_moment),
        float(minimum_margin),
        bool(admissible),
    )


def solve_selected_qss(
    R: float,
    c: float,
    *,
    predecessor_a: float,
    predictor_a: float,
    config: SolverConfig,
) -> QSSResult:
    """Solve one continuation point with adaptive depth doubling."""
    config.validate()
    if R <= 0.0:
        raise ValueError("Canonical QSS solver requires R>0.")
    if not 0.0 <= c <= 1.0:
        raise ValueError("Canonical QSS solver requires 0<=c<=1.")

    if c == 0.0:
        m1 = R / (R + 1.0)
        moments, ratios, margin, minimum, monotonicity, admissible = (
            _moments_and_diagnostics(
                R,
                c,
                R,
                config.initial_depth,
                config.maximum_reported_moment_depth,
            )
        )
        return QSSResult(
            R=R,
            c=c,
            m1=m1,
            a=R,
            moments=moments,
            ratios=ratios,
            accepted_depth=0,
            previous_depth=0,
            depth_doubling_difference=0.0,
            fixed_point_residual=0.0,
            continued_fraction_tail_bound_y1=0.0,
            a_posteriori_root_error_estimate=0.0,
            bessel_m1=m1,
            bessel_cf_difference=0.0,
            simple_root_margin=margin,
            minimum_moment=minimum,
            minimum_monotonicity_margin=monotonicity,
            admissible=admissible,
            continuation_predecessor_a=predecessor_a,
            continuation_predictor_a=predictor_a,
            root_bracket_left=R,
            root_bracket_right=R,
            bracket_expansions=0,
            bracket_source="exact_c0",
        )

    prior_m1: float | None = None
    prior_depth = 0
    center = predictor_a
    depth = config.initial_depth
    last_details: tuple[
        float,
        tuple[float, float],
        int,
        str,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        tuple[float, ...],
        tuple[float, ...],
        float,
        float,
        float,
        bool,
    ] | None = None

    while depth <= config.max_depth:
        a, bracket, expansions, source = _root_at_depth(
            R, c, depth, center, config.root_xtol
        )
        y, derivative = continued_fraction_tail(a, c * R, depth)
        m1 = (R - a) / c
        fixed_point_residual = abs(m1 - y[1] / c)
        bessel_m1 = stable_bessel_y1(a, c * R) / c
        bessel_difference = abs(m1 - bessel_m1)
        (
            moments,
            ratios,
            simple_margin,
            minimum,
            monotonicity,
            admissible,
        ) = _moments_and_diagnostics(
            R,
            c,
            a,
            depth,
            config.maximum_reported_moment_depth,
        )
        depth_difference = (
            float("inf") if prior_m1 is None else abs(m1 - prior_m1)
        )
        last_details = (
            a,
            bracket,
            expansions,
            source,
            y,
            derivative,
            m1,
            fixed_point_residual,
            bessel_difference,
            moments,
            ratios,
            simple_margin,
            minimum,
            monotonicity,
            admissible,
        )
        depth_threshold = (
            config.depth_absolute_tolerance
            + config.depth_relative_tolerance * abs(m1)
        )
        if (
            prior_m1 is not None
            and depth_difference <= depth_threshold
            and fixed_point_residual <= config.fixed_point_residual_tolerance
            and bessel_difference <= config.bessel_crosscheck_tolerance
            and admissible
            and simple_margin > 0.0
        ):
            tail_bound = continued_fraction_y1_bound(a, c * R, depth)
            root_error = (
                fixed_point_residual + tail_bound / c
            ) / simple_margin
            return QSSResult(
                R=R,
                c=c,
                m1=m1,
                a=a,
                moments=moments,
                ratios=ratios,
                accepted_depth=depth,
                previous_depth=prior_depth,
                depth_doubling_difference=depth_difference,
                fixed_point_residual=fixed_point_residual,
                continued_fraction_tail_bound_y1=tail_bound,
                a_posteriori_root_error_estimate=root_error,
                bessel_m1=bessel_m1,
                bessel_cf_difference=bessel_difference,
                simple_root_margin=simple_margin,
                minimum_moment=minimum,
                minimum_monotonicity_margin=monotonicity,
                admissible=admissible,
                continuation_predecessor_a=predecessor_a,
                continuation_predictor_a=predictor_a,
                root_bracket_left=bracket[0],
                root_bracket_right=bracket[1],
                bracket_expansions=expansions,
                bracket_source=source,
            )
        prior_m1 = m1
        prior_depth = depth
        center = a
        depth *= 2

    assert last_details is not None
    a, _, _, _, _, _, m1, residual, bessel_difference, _, _, margin, _, _, admissible = (
        last_details
    )
    raise RuntimeError(
        "Canonical QSS depth adaptation failed: "
        f"R={R}, c={c}, max_depth={config.max_depth}, m1={m1}, a={a}, "
        f"residual={residual}, Bessel difference={bessel_difference}, "
        f"simple margin={margin}, admissible={admissible}."
    )


def solve_continuation(
    R: float,
    c_values: Iterable[float],
    config: SolverConfig | None = None,
) -> list[QSSResult]:
    """Continue the selected nonlinear component from c=0 over a sorted grid."""
    active_config = config or SolverConfig()
    c_array = np.asarray(tuple(float(value) for value in c_values), dtype=float)
    if c_array.ndim != 1 or len(c_array) == 0:
        raise ValueError("c_values must be a nonempty one-dimensional sequence.")
    if abs(c_array[0]) > 1.0e-15:
        raise ValueError("Continuation grid must start at c=0.")
    if np.any(np.diff(c_array) <= 0.0):
        raise ValueError("Continuation c-values must be strictly increasing.")

    results: list[QSSResult] = []
    previous_a = R
    previous_previous_a = R
    previous_c = 0.0
    previous_previous_c = 0.0
    for index, c in enumerate(c_array):
        if index < 2 or previous_c == previous_previous_c:
            predictor_a = previous_a
        else:
            slope = (previous_a - previous_previous_a) / (
                previous_c - previous_previous_c
            )
            predictor_a = previous_a + slope * (c - previous_c)
        predictor_a = min(max(predictor_a, R - c), R)
        result = solve_selected_qss(
            R,
            float(c),
            predecessor_a=previous_a,
            predictor_a=predictor_a,
            config=active_config,
        )
        results.append(result)
        previous_previous_a, previous_a = previous_a, result.a
        previous_previous_c, previous_c = previous_c, float(c)
    return results


def solve_fixed_depth_path(
    R: float,
    c_values: Iterable[float],
    depth: int,
    maximum_moment_depth: int = 10,
) -> list[FixedDepthResult]:
    """Fixed-depth continuation used only for the convergence sidecar."""
    c_array = np.asarray(tuple(float(value) for value in c_values), dtype=float)
    if abs(c_array[0]) > 1.0e-15 or np.any(np.diff(c_array) <= 0.0):
        raise ValueError("Fixed-depth path must start at zero and increase.")
    output: list[FixedDepthResult] = []
    previous_a = R
    for c in c_array:
        if c == 0.0:
            output.append(
                FixedDepthResult(
                    R=R,
                    c=0.0,
                    depth=depth,
                    m1=R / (R + 1.0),
                    a=R,
                    fixed_point_residual=0.0,
                    bessel_cf_difference=0.0,
                    simple_root_margin=1.0,
                    admissible=True,
                    bracket_left=R,
                    bracket_right=R,
                    bracket_expansions=0,
                    bracket_source="exact_c0",
                )
            )
            continue
        a, bracket, expansions, source = _root_at_depth(
            R, float(c), depth, previous_a
        )
        y, derivative = continued_fraction_tail(a, c * R, depth)
        m1 = (R - a) / c
        residual = abs(m1 - y[1] / c)
        bessel_difference = abs(m1 - stable_bessel_y1(a, c * R) / c)
        _, _, margin, _, _, admissible = _moments_and_diagnostics(
            R, float(c), a, depth, maximum_moment_depth
        )
        output.append(
            FixedDepthResult(
                R=R,
                c=float(c),
                depth=depth,
                m1=m1,
                a=a,
                fixed_point_residual=residual,
                bessel_cf_difference=bessel_difference,
                simple_root_margin=margin,
                admissible=admissible,
                bracket_left=bracket[0],
                bracket_right=bracket[1],
                bracket_expansions=expansions,
                bracket_source=source,
            )
        )
        previous_a = a
    return output
