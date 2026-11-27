"""Trajectory sampling and the configurable endpoint-completed pathway index."""

import numpy as np

from common.numerics import adaptive_discrete_sampling, area_under_curve


def pathway_index(p_index, h_index, *, endpoint_complement=True):
    """Return chronological signed trapezoids, completed at I_p=1 by default.

    The endpoint contribution is ``(1 - p_index[-1]) * h_index[-1]``.
    Disabling it returns only the integral along the observed trajectory.
    Neither the trajectory nor the completed result is sorted or clipped.
    """
    if not isinstance(endpoint_complement, (bool, np.bool_)):
        raise TypeError("endpoint_complement must be boolean")
    p, h = np.asarray(p_index, dtype=float), np.asarray(h_index, dtype=float)
    if p.ndim != 1 or h.ndim != 1 or p.shape != h.shape or not p.size:
        raise ValueError("pathway index requires equally sized nonempty one-dimensional trajectories")
    return float(area_under_curve(np.array([p, h]), complement=1.0 if endpoint_complement else None))


def sample_indices(end, point):
    """Return chronological (steps, I_h, I_p, I_s) on the existing adaptive rule.

    point receives an integer completed step and returns (I_p, I_s, I_h).
    All three values participate in refinement, including when only I_w is
    requested. Each trajectory retains its own complete zero-to-end sampling.
    """
    def evaluate(step):
        return np.asarray(point(int(step)), dtype=float)

    def error(left, right, midpoint, fraction):
        residual = np.abs(midpoint - (left * (1 - fraction) + right * fraction))
        return float(np.max(residual, where=np.isfinite(residual), initial=0))

    if end == 0:
        p, s, h = evaluate(0)[:, None]
        return np.array([0]), h, p, s
    steps, values = adaptive_discrete_sampling(
        evaluate, .01, 0, end, max_interval=512, err_func=error, int_midpoint=True,
    )
    p, s, h = np.asarray(values).T
    return np.asarray(steps, dtype=int), h, p, s
