"""Whole-curve release clocks in whichever time units the caller supplies."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import exp1, gammaln


def checked_curve(time, values):
    """Return finite matching 1D arrays; reject fewer than two points or time not starting at zero and strictly increasing."""
    time, values = np.asarray(time, dtype=float), np.asarray(values, dtype=float)
    if time.ndim != 1 or values.shape != time.shape or len(time) < 2:
        raise ValueError("time and values must be matching vectors of length >=2")
    if not np.isfinite(time).all() or not np.isfinite(values).all():
        raise ValueError("curve must be finite")
    if not np.isclose(time[0], 0, atol=1e-12, rtol=0) or np.any(np.diff(time) <= 0):
        raise ValueError("time must start at zero and increase strictly")
    return time, values


@dataclass(frozen=True)
class ReleaseClock:
    """Finite-window release diagnostics in the caller's time units.

    area is the survival integral and t50 its first half-release crossing; horizon
    is the observed endpoint. Weak releases have NaN clocks/survival. status records
    whether the plateau is supplied or estimated, independently of the fitted models.
    """

    plateau: float
    amplitude: float
    horizon: float
    area: float
    t50: float
    rebound: float
    negative_area: float
    status: str
    survival: np.ndarray


def release_clock(
    time, energy, *, plateau: float | None = None, converged: bool = False, amplitude_floor: float = 1e-12
) -> ReleaseClock:
    """Finite-window integral of max((U-Uinf)/(U0-Uinf),0).

    No monotone envelope or upper clipping: rebounds remain in the area.
    If no independent plateau is provided, the tail median is an endpoint
    estimate, not a proof of the infinite-time asymptote. The status preserves
    that distinction even when the solver reports convergence.
    """
    time, energy = checked_curve(time, energy)
    estimated = plateau is None
    if estimated:
        tail = min(len(energy), min(501, max(51, len(energy) // 10)))
        # Short smoke traces do not have enough tail points to estimate a plateau.
        tail = min(tail, max(1, len(energy) // 5))
        plateau = float(np.median(energy[-tail:]))
    if not np.isfinite(plateau):
        raise ValueError("plateau must be finite")
    amplitude = float(energy[0] - plateau)
    if amplitude <= amplitude_floor:
        return ReleaseClock(
            float(plateau),
            amplitude,
            float(time[-1]),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            "weak_release",
            np.full_like(energy, np.nan),
        )
    raw = (energy - plateau) / amplitude
    survival = np.maximum(raw, 0)
    hits = np.flatnonzero(raw <= 0.5)
    t50 = np.nan
    if len(hits):
        i = int(hits[0])
        t50 = (
            float(time[i])
            if i == 0
            else float(time[i - 1] + (time[i] - time[i - 1]) * (raw[i - 1] - 0.5) / (raw[i - 1] - raw[i]))
        )
    status = ("converged_tail_estimate" if converged else "endpoint_estimate") if estimated else "known_plateau"
    return ReleaseClock(
        float(plateau),
        amplitude,
        float(time[-1]),
        float(trapezoid(survival, time)),
        t50,
        float(np.maximum(np.diff(raw), 0).max(initial=0)),
        float(trapezoid(np.maximum(-raw, 0), time)),
        status,
        survival,
    )


def gompertz_survival(time, initial_hazard: float, acceleration: float):
    """Return survival for hazard h(t)=initial_hazard*exp(acceleration*t); zero acceleration gives exponential decay."""
    if not np.isfinite([initial_hazard, acceleration]).all() or initial_hazard <= 0 or acceleration < 0:
        raise ValueError("positive hazard and nonnegative acceleration required")
    time = np.asarray(time, dtype=float)
    if np.any(time < 0) or not np.isfinite(time).all():
        raise ValueError("time must be finite and nonnegative")
    if acceleration == 0:
        return np.exp(-initial_hazard * time)
    with np.errstate(over="ignore"):
        return np.exp(-initial_hazard / acceleration * np.expm1(acceleration * time))


def gompertz_mean(initial_hazard: float, acceleration: float) -> float:
    """Return the infinite Gompertz survival integral in the reciprocal hazard units, using a stable large-ratio expansion."""
    gompertz_survival([0.0], initial_hazard, acceleration)
    if acceleration == 0:
        return 1 / initial_hazard
    z = initial_hazard / acceleration
    if z > 50:
        # exp(z)*E1(z) asymptotic expansion avoids overflow; 14 terms at z>50.
        term, series = 1.0, 1.0
        for n in range(1, 15):
            term *= -n / z
            series += term
        return float(series / initial_hazard)
    return float(np.exp(z) * exp1(z) / acceleration)


def stretched_mean(rate: float, shape: float) -> float:
    """Return the infinite integral of exp(-(rate*t)**shape); both parameters must be finite and positive."""
    if not np.isfinite([rate, shape]).all() or rate <= 0 or shape <= 0:
        raise ValueError("rate and shape must be finite and positive")
    return float(np.exp(gammaln(1 + 1 / shape) - np.log(rate)))


def mixture_mean(weight: float, fast_mean: float, node_mean: float) -> float:
    """Return the weighted infinite component means; weight is the fast-component fraction in [0,1]."""
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("mixture weight must lie in [0,1]")
    if not np.isfinite([fast_mean, node_mean]).all() or min(fast_mean, node_mean) <= 0:
        raise ValueError("component means must be finite and positive")
    return float(weight * fast_mean + (1 - weight) * node_mean)
