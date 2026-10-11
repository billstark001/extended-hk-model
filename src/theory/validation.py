"""Matched node/edge field and energy-trajectory comparisons."""

import math

import numpy as np

from common.numerics.distributions import hellinger, rebin_matrix


def fields_on(frame, b):
    """Rebin one frame to b cells, returning (rho, edge); frame edge may be flattened."""
    p = np.asarray(frame["rho"])
    old = len(p)
    e = np.asarray(frame["edge"]).reshape(old, old)
    if old != b:
        r = rebin_matrix(old, b)
        p = r @ p
        e = r @ e @ r.T
    return p, e


def compare_fields(a, b):
    """Return node W1 and node/edge Hellinger distances after conservative transfer to the coarser [-1,1] grid."""
    count = min(len(a["rho"]), len(b["rho"]))
    p, e = fields_on(a, count)
    q, f = fields_on(b, count)
    return {
        "node_hellinger": hellinger(p, q),
        "edge_hellinger": hellinger(e, f),
        "node_w1": float(np.abs(np.cumsum(p / p.sum() - q / q.sum())).sum() * 2 / count),
    }


def trajectory_gap(a, b, name):
    """Return time-averaged absolute field gap, its normalization by the first trajectory initial value, and initial_match; horizons must match."""
    ta = np.array([f["time"] for f in a])
    tb = np.array([f["time"] for f in b])
    if not math.isclose(ta[-1], tb[-1], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("paired horizons differ")
    t = np.unique(np.r_[ta, tb])
    ya = np.interp(t, ta, [f[name] for f in a])
    yb = np.interp(t, tb, [f[name] for f in b])
    if not math.isclose(ya[0], yb[0], rel_tol=1e-9, abs_tol=1e-12):
        # ER row feasibility can change initial edge energy. Normalize by micro initial.
        same = False
    else:
        same = True
    norm = abs(ya[0])
    raw = float(np.trapezoid(np.abs(ya - yb), t) / max(t[-1] - t[0], 1e-15))
    return {"gap": raw / norm if norm > 1e-14 else None, "raw_gap": raw, "initial_match": same}
