"""Whole-curve mean clocks and paired resampling, in the supplied time units."""

import numpy as np

from .means import checked_curve, release_clock


def plateau_check(time, energy, *, window=20.0, minimum_horizon=120.0, tolerance=1e-4):
    """Confirm low variation in each of two disjoint terminal time windows.

    This finite-window diagnostic is independent of solver stopping. It does not
    establish an infinite-time asymptote, and short traces cannot pass it.
    """
    time, energy = checked_curve(time, energy)
    if time[-1] < max(minimum_horizon, 2 * window):
        return False
    scale = max(abs(energy[0]), 1e-12)
    for lo, hi in ((time[-1] - 2 * window, time[-1] - window), (time[-1] - window, time[-1])):
        values = energy[(time >= lo) & (time <= hi)]
        if len(values) < 3 or np.ptp(values) > tolerance * scale:
            return False
    return True


def clock_statistics(time, node, edge, weights, *, confidence=0.95, plateaus=None):
    """Integral clocks of mean curves, paired bootstrap ratio and uncertainty.

    node/edge are branch-by-time matrices. plateaus=None re-estimates each
    bootstrap curve's endpoint plateau; supplied plateaus are held fixed.
    Samples with weak releases remain invalid and their fraction is reported.
    """
    node, edge, weights = map(lambda v: np.asarray(v, float), (node, edge, weights))
    if node.ndim != 2 or edge.shape != node.shape or weights.ndim != 2 or weights.shape[1] != len(node):
        raise ValueError("matching branch curves and resampling weights required")
    if (
        not 0 < confidence < 1
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.allclose(weights.sum(1), 1)
    ):
        raise ValueError("invalid confidence or resampling weights")
    checked_curve(time, node[0])
    checked_curve(time, edge[0])
    points, boots = {}, {}
    for name, matrix in (("node", node), ("edge", edge)):
        plateau = None if plateaus is None else plateaus[name]
        points[name] = release_clock(time, matrix.mean(0), plateau=plateau)
        boots[name] = np.array([release_clock(time, c, plateau=plateau).area for c in weights @ matrix])
    valid = np.isfinite(boots["node"]) & np.isfinite(boots["edge"]) & (boots["node"] > 0)
    ratio = np.divide(boots["edge"], boots["node"], out=np.full(len(weights), np.nan), where=valid)
    out = dict(horizon=float(time[-1]), plateau_reference="own_endpoint" if plateaus is None else "fixed_pair_endpoint")
    tail = (1 - confidence) / 2
    for name, sample, point in (
        ("node", boots["node"], points["node"].area),
        ("edge", boots["edge"], points["edge"].area),
        ("edge_to_node", ratio, points["edge"].area / points["node"].area if points["node"].area > 0 else np.nan),
    ):
        finite = sample[np.isfinite(sample)]
        lo, hi = np.quantile(finite, [tail, 1 - tail]) if len(finite) else (np.nan, np.nan)
        out[name + "_mean"] = float(point)
        out[name + "_lower"] = float(lo) if len(node) > 1 else None
        out[name + "_upper"] = float(hi) if len(node) > 1 else None
        out[name + "_valid_fraction"] = len(finite) / len(sample)
    for name, clock in points.items():
        out.update(
            {name + "_" + k: getattr(clock, k) for k in ("plateau", "amplitude", "status", "rebound", "negative_area")}
        )
    return out, {
        name: sample for name, sample in (("node", boots["node"]), ("edge", boots["edge"]), ("edge_to_node", ratio))
    }
