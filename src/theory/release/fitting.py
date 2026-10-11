"""Parametric survival comparisons and curve-error diagnostics in scaled time."""

import json

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaincc


def model(t, p, name):
    """Evaluate the selected survival family using its log-parameter representation."""
    if name == "weibull2":
        scale, beta = np.exp(p)
        return np.exp(-np.minimum((t / scale) ** beta, 745))
    if name == "gamma2":
        scale, shape = np.exp(p)
        return gammaincc(shape, t / scale)
    if name == "generalized_gamma3":
        scale, beta, shape = np.exp(p)
        return gammaincc(shape, (t / scale) ** beta)
    if name == "hazard3":
        h0, hinf, tc = np.exp(p)
        H = hinf * t + (h0 - hinf) * tc * -np.expm1(-t / tc)
        return np.exp(-np.maximum(H, 0))
    if name == "gompertz2":
        h, g = np.exp(p)
        H = h / g * np.expm1(np.minimum(g * t, 30))
        return np.exp(-np.minimum(H, 745))
    raise ValueError(name)


def fit_model(c, name):
    """Fit the requested parametric survival family to one normalized case; return predictions and fit diagnostics."""
    from .curves import quantiles

    idx = np.unique(np.linspace(0, len(c["t"]) - 1, min(1501, len(c["t"]))).astype(int))
    t = c["t"][idx]
    y = c["se"][idx]
    mid = quantiles(t, y, [0.5])[0]
    x = t / mid
    if name in ("weibull2", "gamma2"):
        starts = [[1.0, b] for b in (0.5, 1, 2, 5)]
        lo = [0.001, 0.05]
        hi = [1000, 100]
    elif name == "generalized_gamma3":
        starts = [[1.0, b, a] for b, a in ((1, 1), (0.5, 2), (2, 0.5), (1, 5), (2, 2), (0.5, 0.5))]
        lo = [0.0001, 0.08, 0.03]
        hi = [10000.0, 15, 100]
    elif name == "hazard3":
        starts = [[a, b, c] for a, b, c in ((0.2, 2, 1), (2, 0.2, 1), (0.7, 0.7, 1), (0.1, 4, 3), (3, 0.1, 0.1))]
        lo = [1e-07, 1e-07, 0.0001]
        hi = [10000.0, 10000.0, 10000.0]
    else:
        starts = [[0.3, 0.5], [0.05, 2], [0.7, 0.01]]
        lo = [1e-08, 1e-08]
        hi = [1000.0, 1000.0]
    fits = [
        least_squares(
            lambda p: model(x, p, name) - y,
            np.log(p),
            bounds=(np.log(lo), np.log(hi)),
            max_nfev=400,
            ftol=1e-09,
            xtol=1e-09,
            gtol=1e-09,
        )
        for p in starts
    ]
    fit = min(fits, key=lambda f: f.cost)
    return (
        model(c["t"] / mid, fit.x, name),
        dict(params=json.dumps(np.exp(fit.x).tolist()), time_scale=mid, success=bool(fit.success)),
    )


def errors(t, y, p):
    """Return full/active-window and pointwise errors for a prediction on the supplied time grid."""
    err = p - y
    active = y > 0.01
    return dict(
        rmse=float(np.sqrt(np.mean(err**2))),
        max_error=float(np.max(abs(err))),
        active_rmse=float(np.sqrt(np.mean(err[active] ** 2))),
    )


def fit_clock_ratio(x, y, *, softplus=False):
    """Two-parameter illustrative log-ratio regression; return parameters/predictions."""
    if not softplus:
        design = np.c_[np.ones(len(x)), -x]
        p = np.linalg.lstsq(design, y, rcond=None)[0]
        return p, design @ p
    fit = least_squares(lambda p: -np.logaddexp(0, np.exp(p[1]) * (x - p[0])) - y, [np.log(6), 0])
    p = [float(np.exp(fit.x[0])), float(np.exp(fit.x[1]))]
    return p, -np.logaddexp(0, p[1] * (x - np.log(p[0])))
