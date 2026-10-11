"""Finite exposure-list convolutions used to audit the runtime closure."""

import math

import numpy as np


def pmf(n, p):
    """Return binomial masses for counts 0 through n in that order."""
    return np.array([math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)])


def deposit(axis, positions, weights):
    """Deposit weighted positions linearly onto a uniform axis, clamping positions beyond the endpoint centers."""
    row = np.zeros(len(axis))
    u = np.clip((np.asarray(positions) - axis[0]) / (axis[1] - axis[0]), 0, len(axis) - 1)
    lower = np.floor(u).astype(int)
    upper = np.minimum(lower + 1, len(axis) - 1)
    frac = u - lower
    np.add.at(row, lower, weights * (1 - frac))
    np.add.at(row, upper, weights * frac)
    return row


def powers(p, count):
    """Return convolution powers 0 through count of a one-dimensional discrete mass function."""
    out = [np.array([1.0])]
    for _ in range(count):
        out.append(np.convolve(out[-1], p))
    return out


def exact_row(frozen, meta, source):
    """Audit one HK source row by exact finite-list count/sum convolutions.

    frozen supplies the axis and flattened neighbor/recommendation kernels; meta
    supplies epsilon, alpha, h, degree and recommendations. Return the deposited
    transition row, unscaled increment mean/second moment, empty-list probability
    and Gaussian tail proxy. Only deposition applies alpha*h to increments.
    """
    axis = np.asarray(frozen["axis"])
    b = len(axis)
    eps = meta["epsilon"]
    alpha = meta["alpha"] * meta["h"]
    n, m = meta["degree"], meta["recommendations"]
    mask = np.abs(axis - axis[source]) <= eps + 1e-14
    p = np.asarray(frozen["neighbor_kernel"]).reshape(b, b)[source] * mask
    q = np.asarray(frozen["recommendation_kernel"]).reshape(b, b)[source] * mask
    pn, pq = p.sum(), q.sum()
    a = p / pn if pn else np.eye(1, b, source)[0]
    c = q / pq if pq else np.eye(1, b, source)[0]
    pa, pc = powers(a, n), powers(c, m)
    bn, bm = pmf(n, pn), pmf(m, pq)
    row = np.zeros(b)
    tail = 0.0
    mean = 0.0
    second = 0.0
    empty = (1 - pn) ** n * (1 - pq) ** m
    for j in range(n + 1):
        for k in range(m + 1):
            prob = bn[j] * bm[k]
            total = j + k
            if not total:
                row[source] += prob
                continue
            sums = np.convolve(pa[j], pc[k])
            average = axis[0] + np.arange(len(sums)) * (axis[1] - axis[0]) / total
            inc = average - axis[source]
            mean += prob * np.dot(sums, inc)
            second += prob * np.dot(sums, inc * inc)
            row += deposit(axis, axis[source] + alpha * inc, prob * sums)
            mu = (j * np.dot(a, axis - axis[source]) + k * np.dot(c, axis - axis[source])) / total
            var = (
                j * np.dot(a, (axis - axis[source] - np.dot(a, axis - axis[source])) ** 2)
                + k * np.dot(c, (axis - axis[source] - np.dot(c, axis - axis[source])) ** 2)
            ) / total**2
            if var > 1e-18:
                lo = (axis[mask][0] - axis[source] - mu) / math.sqrt(var)
                hi = (axis[mask][-1] - axis[source] - mu) / math.sqrt(var)
                tail += prob * (0.5 * (1 + math.erf(lo / math.sqrt(2))) + 0.5 * math.erfc(hi / math.sqrt(2)))
    return row, mean, second, empty, tail
