"""Reduced and alternative empirical edge mixture fitting kernels."""

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def fixed_exponent(c, t, n, y, beta):
    def objective(logk, returnw=False):
        f = np.exp(-((np.exp(logk) * c["r"] * t) ** beta))
        d = f - n
        w = float(np.clip((y - n) @ d / max(d @ d, 1e-30), 0, 1))
        cost = np.mean((n + w * d - y) ** 2)
        return (cost, w) if returnw else cost

    grid = np.linspace(np.log(1e-4), np.log(1e3), 101)
    costs = np.array([objective(k) for k in grid])
    local = [j for j in range(1, len(grid) - 1) if costs[j] <= costs[j - 1] and costs[j] <= costs[j + 1]]
    candidates = [(costs[0], grid[0]), (costs[-1], grid[-1])]
    for j in local:
        fit = minimize_scalar(objective, bounds=(grid[j - 1], grid[j + 1]), method="bounded", options={"xatol": 1e-10})
        candidates.append((fit.fun, fit.x))
    cost, logk = min(candidates)
    _, w = objective(logk, True)
    k = np.exp(logk)
    pred = w * np.exp(-((k * c["r"] * c["t"]) ** beta)) + (1 - w) * c["sn"]
    return w, k, pred


def alternative_branch(tt, sn, se, r, limit):
    starts = []
    ks = np.linspace(np.log(1e-4), np.log(1e3), 101)

    def shape(lk, lb):
        return np.exp(-np.exp(np.minimum(np.exp(lb) * np.log(np.maximum(np.exp(lk) * r * tt, 1e-300)), 7)))

    for b in (0.3, 0.5, 1.0, 1.5, 2.5, 4.5, 10.0, 25.0, 49.0):
        if b > limit:
            continue
        best = None
        for lk in ks:
            d = shape(lk, np.log(b)) - sn
            w = float(np.clip((se - sn) @ d / max(d @ d, 1e-30), 0, 1))
            cost = np.mean((sn + w * d - se) ** 2)
            if best is None or cost < best[0]:
                best = (cost, [w, lk, np.log(b)])
        starts.append(best[1])

    def residual(p):
        return sn + p[0] * (shape(p[1], p[2]) - sn) - se

    fits = [
        least_squares(
            residual,
            p,
            bounds=([0, np.log(1e-4), np.log(0.2)], [1, np.log(1e3), np.log(limit)]),
            max_nfev=800,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        for p in starts
    ]
    fit = min(fits, key=lambda f: f.cost)
    return fit
