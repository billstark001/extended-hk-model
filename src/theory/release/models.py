"""Analytic fitted survival functions and parameter-derived release quantiles."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import exp1

from .curves import quantiles


def scaled_e1(x):
    """Evaluate exp(x)*E1(x) stably, using a large-x asymptotic expansion above 50."""
    if x <= 50:
        return float(np.exp(x) * exp1(x))
    term = 1.0
    total = 1.0
    for i in range(1, 15):
        term *= -i / x
        total += term
    return total / x


def sn(t, h, g):
    """Evaluate Gompertz node survival with positive initial hazard h and acceleration g, clipping overflow-prone exponentials."""
    return np.exp(-np.minimum(h / g * np.expm1(np.minimum(g * np.asarray(t), 700)), 745))


def times_node(p, h, g):
    """Return Gompertz release quantiles for probabilities p in (0,1), in the fit time units."""
    return np.log1p(g * -np.log1p(-np.asarray(p)) / h) / g


def times_full(p, h, g, w, lam, beta):
    """Solve one release quantile of the fast KWW plus Gompertz node mixture by bracketing component quantiles."""
    tn = float(times_node(p, h, g))
    tf = float((-np.log1p(-p)) ** (1 / beta) / lam)
    return brentq(
        lambda t: w * np.exp(-((lam * t) ** beta)) + (1 - w) * float(sn(t, h, g)) - (1 - p),
        0,
        max(tn, tf) * (1 + 1e-10),
        xtol=1e-11,
    )


def observed_q(t, s, p):
    """Return the historical interpolated observed quantile, or NaN when the sampled curve never reaches it."""
    if min(s) > 1 - p:
        return float("nan")
    return float(quantiles(t, s, [p])[0])


def truncated_node(horizon, h, g):
    """Numerical finite-window integral of fitted Gompertz survival."""
    from scipy.integrate import quad

    return quad(lambda u: float(sn(u, h, g)), 0, horizon, epsabs=1e-9)[0]


def truncated_fast(horizon, rate, beta):
    """Numerical finite-window integral of fitted stretched-exponential survival."""
    from scipy.integrate import quad

    return quad(lambda u: float(np.exp(-((rate * u) ** beta))), 0, horizon, epsabs=1e-9)[0]
