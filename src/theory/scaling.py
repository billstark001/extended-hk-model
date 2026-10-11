"""Curve comparison for tau=alpha*t, r=q/alpha, without endpoint alignment."""

import numpy as np
from scipy.integrate import trapezoid

from .release import checked_curve


def curve_distance(reference_time, reference, time, values) -> dict[str, float]:
    """Compare curves on the union grid up to their common horizon.

    Both times must start at zero and use the same units. Return maximum and RMS
    gaps scaled by the reference range or initial magnitude, without endpoint
    normalization or rescaling either clock.
    """
    t0, y0 = checked_curve(reference_time, reference)
    t1, y1 = checked_curve(time, values)
    horizon = min(t0[-1], t1[-1])
    grid = np.unique(np.r_[t0[t0 <= horizon], t1[t1 <= horizon], horizon])
    a, b = np.interp(grid, t0, y0), np.interp(grid, t1, y1)
    scale = max(float(np.ptp(a)), abs(float(a[0])), 1e-15)
    delta = a - b
    return {
        "horizon": float(horizon),
        "absolute_max": float(np.max(abs(delta))),
        "relative_max": float(np.max(abs(delta)) / scale),
        "relative_rms": float(np.sqrt(trapezoid(delta**2, grid) / horizon) / scale),
    }
