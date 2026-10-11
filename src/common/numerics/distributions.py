"""Distribution distances and conservative interval rebinning."""

import numpy as np


def hellinger(p, q):
    """Normalize nonnegative-clipped masses and return their Hellinger distance, or None if either total is zero."""
    p = np.maximum(np.asarray(p, float), 0)
    q = np.maximum(np.asarray(q, float), 0)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p / p.sum()) - np.sqrt(q / q.sum())) ** 2)))


def rebin_matrix(old, new, lower=-1.0, upper=1.0):
    """Return a (new, old) conservative mass-transfer matrix for uniform bins.

    Both grids span [lower, upper]. Apply matrix @ mass to a mass vector, not a
    density; an edge-mass matrix transfers as matrix @ edge @ matrix.T.
    """
    a = np.linspace(lower, upper, old + 1)
    b = np.linspace(lower, upper, new + 1)
    return np.maximum(0, np.minimum(b[1:, None], a[None, 1:]) - np.maximum(b[:-1, None], a[None, :-1])) / (
        (upper - lower) / old
    )
