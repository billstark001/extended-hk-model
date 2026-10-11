"""First-crossing clocks with explicit finite-horizon status."""

import math

import numpy as np


def crossing(t, values, threshold):
    """Return the first downward threshold crossing by linear interpolation, or None if the finite window has none."""
    hits = np.flatnonzero(values <= threshold)
    if not len(hits):
        return None
    i = int(hits[0])
    if i == 0:
        return float(t[0])
    d = values[i - 1] - values[i]
    return float(t[i - 1] + (t[i] - t[i - 1]) * (values[i - 1] - threshold) / d) if d > 0 else float(t[i])


def clocks(frames, converged, name, tail=0.1):
    """Return tail-median T10/T50/T90 first crossings for one frame field.

    Times retain frame time units. tail is the terminal sample fraction; converged
    marks plateau verification, but finite crossings may still be reported without
    it. Weak release and nonmonotonicity remain explicit statuses.
    """
    t = np.asarray([f["time"] for f in frames])
    y = np.asarray([f[name] for f in frames])
    c = float(np.median(y[max(0, len(y) - max(3, math.ceil(tail * len(y)))) :]))
    amp = float(y[0] - c)
    out = {
        "plateau": c,
        "amplitude": amp,
        "plateau_verified": bool(converged),
        "nonmonotone_release": False,
        "T10": None,
        "T50": None,
        "T90": None,
        "status": "no_plateau",
    }
    if amp <= max(1e-12, abs(y[0]) * 1e-5):
        out["status"] = "zero_or_negative_drop"
        return out
    out["nonmonotone_release"] = bool(np.maximum(np.diff(y), 0).sum() > 0.05 * amp)
    out["status"] = (
        "first_crossing_nonmonotone" if out["nonmonotone_release"] else "ok" if converged else "finite_horizon_only"
    )
    for p in (0.1, 0.5, 0.9):
        out[f"T{round(100 * p)}"] = crossing(t, y, c + (1 - p) * amp)
    return out


def clock_pair(trace):
    """Return both energy-channel clocks, tail sensitivity and verified log(edge/node) ratios from a recorded trace."""
    verified = bool(trace.get("diagnostics", {}).get("converged", False))
    out = {}
    for channel, name in (("rho", "node_energy"), ("edge", "edge_energy")):
        c = clocks(trace["frames"], verified, name)
        sensitivity = clocks(trace["frames"], verified, name, 0.2)
        out.update({f"{channel}_{k}": v for k, v in c.items()})
        out[f"{channel}_plateau_tail20"] = sensitivity["plateau"]
        out[f"{channel}_T50_tail20"] = sensitivity["T50"]
    for p in (10, 50, 90):
        a, b = out[f"rho_T{p}"], out[f"edge_T{p}"]
        out[f"delta{p}"] = math.log(b / a) if verified and a is not None and b is not None and a > 0 and b > 0 else None
    out["horizon"] = trace["frames"][-1]["time"]
    out["order_consistent"] = None
    signs = [np.sign(out[f"delta{p}"]) for p in (10, 50, 90) if out[f"delta{p}"] is not None]
    if len(signs) == 3:
        out["order_consistent"] = len(set(signs)) == 1
    return out


def batch_half_times(times, curves, tail):
    """Return interpolated T50 for each row of curves; absent crossings or weak amplitudes become NaN."""
    curves = np.atleast_2d(curves)
    count = max(3, math.ceil(tail * curves.shape[1]))
    plateau = np.median(curves[:, -count:], axis=1)
    amplitude = curves[:, 0] - plateau
    threshold = plateau + 0.5 * amplitude
    hits = curves <= threshold[:, None]
    index = hits.argmax(axis=1)
    previous = np.maximum(index - 1, 0)
    row = np.arange(len(curves))
    drop = curves[row, previous] - curves[row, index]
    frac = np.divide(curves[row, previous] - threshold, drop, out=np.zeros(len(curves)), where=drop > 0)
    result = times[previous] + (times[index] - times[previous]) * frac
    valid = hits.any(axis=1) & (amplitude > np.maximum(1e-12, abs(curves[:, 0]) * 1e-5))
    return np.where(valid, result, np.nan)


def curve_clock_interval(times, curves, weights, protocol):
    """Bootstrap T50 of the mean energy curve, using caller-supplied resampling weights.

    Return a clock/interval dict, mean curve and pointwise confidence band. This is
    not the mean of individual T50 values; valid_bootstrap_fraction reports missing
    bootstrap crossings. Units follow times and the plateau uses protocol tail_fractions.
    """
    curves = np.asarray(curves)
    mean = curves.mean(axis=0)
    boot = weights @ curves
    half = batch_half_times(times, boot, protocol["tail_fractions"][0])
    tail = (1 - protocol["confidence_level"]) / 2
    finite = half[np.isfinite(half)]
    value = batch_half_times(times, mean, protocol["tail_fractions"][0])[0]
    low, high = np.quantile(finite, (tail, 1 - tail)) if len(finite) else (None, None)
    return (
        {
            "T50_of_mean_curve": float(value) if np.isfinite(value) else None,
            "lower": float(low) if low is not None and len(curves) > 1 else None,
            "upper": float(high) if high is not None and len(curves) > 1 else None,
            "valid_bootstrap_fraction": len(finite) / len(half),
        },
        mean,
        np.quantile(boot, (tail, 1 - tail), axis=0),
    )
