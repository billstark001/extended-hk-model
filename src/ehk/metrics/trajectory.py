"""Trajectory sampling shared by microscopic measurement paths."""

import numpy as np

from common.numerics import adaptive_discrete_sampling


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
