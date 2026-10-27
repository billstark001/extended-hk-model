"""Commitment bounds on a declared observation grid, with possible recrossings."""

import numpy as np


def timing(times, probabilities, *, delta=0.1):
    """Bound the first threshold hit on this grid under simultaneous probability bounds.

    Missing future observations give an infinite upper time. A first crossing
    is not a persistence claim. These are not continuous-time entrance bounds.
    """
    tt = np.asarray(times, float)
    if not 0 < delta < 0.5 or tt.ndim != 1 or not len(tt) or not np.isfinite(tt).all() or np.any(tt < 0):
        raise ValueError("nonnegative finite times and 0<delta<1/2 required")
    if len(tt) != len(probabilities) or np.any(np.diff(tt) <= 0):
        raise ValueError("one probability result per strictly increasing time required")
    threshold = 1 - delta
    possible = np.array([max(p["upper"]) >= threshold for p in probabilities])
    confirmed = np.array([max(p["lower"]) >= threshold for p in probabilities])
    below = np.array([max(p["upper"]) < threshold for p in probabilities])
    point = np.array([max(p["estimate"]) >= threshold for p in probabilities])

    def first(mask):
        return float(tt[np.flatnonzero(mask)[0]]) if mask.any() else float("inf")

    recrossings = sum(bool(below[i] and confirmed[:i].any()) for i in range(len(tt)))
    return dict(
        delta=delta,
        first_possible_tau=first(possible),
        first_confirmed_tau=first(confirmed),
        first_estimated_tau=first(point),
        confirmed_returns_below=recrossings,
        observed_through_tau=float(tt[-1]),
        time_object="first hit on observation grid",
    )


def ensemble_cdf(rows, times):
    """Equal-initial-state CDF bounds, averaging mother paths within each state."""
    if not rows:
        raise ValueError("nonempty initial-state groups required")
    groups = sorted({r["initial"] for r in rows})
    output = []
    for t in times:
        lo, hi = [], []
        for group in groups:
            selected = [r for r in rows if r["initial"] == group]
            lo.append(np.mean([r["first_confirmed_tau"] <= t for r in selected]))
            hi.append(np.mean([r["first_possible_tau"] <= t or t > r["observed_through_tau"] for r in selected]))
        output.append(
            dict(tau=float(t), cdf_lower=float(np.mean(lo)), cdf_upper=float(np.mean(hi)), initials=len(groups))
        )
    return output
