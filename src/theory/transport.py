"""Synchronous transport decomposition under a specified endpoint transition."""

import numpy as np


def synchronous_increment(edge, replacement, transition, q_step):
    """Return motion, rewiring and mixed increments separately.

    With T(F)=M.T@F@M, E+ - E=(T-I)E + q_step*J + q_step*(T-I)J.
    M is an assumed conditionally independent endpoint transition; this algebra
    does not assert independence of actual microscopic endpoint updates.
    """
    edge, flux, m = map(lambda x: np.asarray(x, dtype=float), (edge, replacement, transition))
    if edge.ndim != 2 or edge.shape[0] != edge.shape[1] or flux.shape != edge.shape or m.shape != edge.shape:
        raise ValueError("square edge, flux and transition arrays must match")
    if not all(np.isfinite(x).all() for x in (edge, flux, m)):
        raise ValueError("arrays must be finite")
    if np.any(m < 0) or not np.allclose(m.sum(1), 1) or not 0 <= q_step <= 1:
        raise ValueError("transition rows must sum to one and q_step must lie in [0,1]")
    return m.T @ edge @ m - edge, q_step * flux, q_step * (m.T @ flux @ m - flux)
