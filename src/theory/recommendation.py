"""Weighted iid candidate laws used by the pair closure; no ranked top-k law."""

import numpy as np


def recommendation_kernel(state, axis, *, kind="random", steepness=1.0, tolerance=0.4, random_ratio=0.0):
    """Return a (bins, bins) row-stochastic candidate-mass kernel.

    Rows condition on source opinion; empty weighted rows fall back to rho.
    random_ratio mixes the weighted law with independent sampling from rho.
    Supported weighted laws are opinion_random and structure_random_l0.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.shape != state.rho.shape or not np.isfinite(axis).all():
        raise ValueError("axis must match rho")
    if (
        not np.isfinite([steepness, tolerance, random_ratio]).all()
        or steepness <= 0
        or tolerance <= 0
        or not 0 <= random_ratio <= 1
    ):
        raise ValueError("invalid recommendation parameters")
    random = np.broadcast_to(state.rho, state.edge.shape).copy()
    if kind == "random":
        return random
    if kind == "opinion_random":
        score = np.maximum(1 - abs(axis[:, None] - axis[None, :]) / tolerance, 0)
    elif kind == "structure_random_l0":
        neighbors = state.neighbors()
        score = neighbors @ neighbors.T
    else:
        raise ValueError("unknown weighted pair recommender")
    weighted = score**steepness * state.rho[None, :]
    totals = weighted.sum(1)
    weighted = np.divide(weighted, totals[:, None], out=random.copy(), where=totals[:, None] > 0)
    return (1 - random_ratio) * weighted + random_ratio * random
