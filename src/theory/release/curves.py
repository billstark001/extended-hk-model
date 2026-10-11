"""Historical survival normalization and interpolation conventions."""

import numpy as np
from scipy.interpolate import PchipInterpolator


def norm(y):
    """Return (tail-normalized survival, plateau, amplitude) under the historical sampled-tail convention."""
    c = np.median(y[-min(501, max(51, len(y) // 10)) :])
    a = y[0] - c
    return ((y - c) / a, c, a)


def quantiles(t, s, levels):
    """Interpolate clipped monotone release progress; missing levels clamp to endpoint times under this historical fit convention."""
    p = np.maximum.accumulate(np.clip(1 - s, 0, 1))
    unique, idx = np.unique(p, return_index=True)
    return np.interp(levels, unique, t[idx])


def qpredict(t, levels, qt):
    """Interpolate cumulative hazard through supplied release quantiles and continue with a positive linear hazard tail."""
    tt = np.r_[0, qt]
    hh = np.r_[0, -np.log1p(-levels)]
    f = PchipInterpolator(tt, hh, extrapolate=False)
    end = float(f.derivative()(tt[-1]))
    end = max(end, 1e-12)
    H = f(np.minimum(t, tt[-1])) + end * np.maximum(t - tt[-1], 0)
    return np.exp(-H)


def interpolate_times(r_train, q_train, r):
    # Positive increments guarantee ordered quantiles at every intermediate r.
    """Interpolate ordered quantile times across positive rate ratios using PCHIP on log-ratio/log-increments; sum the positive increments."""
    increments = np.diff(np.c_[np.zeros(len(q_train)), q_train], axis=1)
    return np.cumsum(np.exp(PchipInterpolator(np.log(r_train), np.log(increments), axis=0)(np.log(r))))


def log_rate_crossings(log_rates, values):
    """Positive rate roots on adjacent sign-changing intervals of a PCHIP fit."""
    from scipy.optimize import brentq

    f = PchipInterpolator(log_rates, values)
    return [
        float(np.exp(brentq(f, log_rates[i], log_rates[i + 1])))
        for i in range(len(log_rates) - 1)
        if values[i] * values[i + 1] < 0
    ]
