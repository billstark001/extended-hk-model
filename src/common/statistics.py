"""Resampling primitives; experiment modules choose independent sampling units."""

import numpy as np


def bootstrap_mean(values, seed=20261006):
    """Return the sample mean and a seeded 95% percentile interval from 2,000 resamples.

    An empty sample has no mean; fewer than two values have no interval. Callers
    must supply independent statistical units rather than correlated branches.
    """
    x = np.asarray(values, float)
    if not len(x):
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    if len(x) == 1:
        return {"n": 1, "mean": float(x[0]), "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(2000, len(x)))].mean(axis=1)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
    }


def bootstrap_weights(count, protocol):
    """Return a (bootstrap_samples, count) matrix of normalized multinomial weights using the protocol seed."""
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    return rng.multinomial(count, np.full(count, 1 / count), size=protocol["bootstrap_samples"]) / count


def mean_interval(values, weights, confidence):
    """Return the mean and percentile interval of weights @ values.

    confidence is a fraction in (0, 1). If any value is nonfinite, report the mean
    of finite entries and valid_processes, but suppress the interval.
    """
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    if not valid.all():
        return {
            "mean": float(values[valid].mean()) if valid.any() else None,
            "lower": None,
            "upper": None,
            "valid_processes": int(valid.sum()),
        }
    tail = (1 - confidence) / 2
    boot = weights @ values
    low, high = np.quantile(boot, (tail, 1 - tail))
    return {
        "mean": float(values.mean()),
        "lower": float(low) if len(values) > 1 else None,
        "upper": float(high) if len(values) > 1 else None,
        "valid_processes": len(values),
    }


def hierarchical_weights(cluster_sizes, protocol):
    """Resample independent initial states, then process branches within each.

    Each initial state has equal weight even with unequal branch counts.
    The returned weights act on concatenated branch arrays in cluster order.
    """
    sizes = np.asarray(cluster_sizes, int)
    if sizes.ndim != 1 or not len(sizes) or np.any(sizes < 1):
        raise ValueError("positive cluster sizes required")
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    count = protocol["bootstrap_samples"]
    offsets = np.r_[0, sizes.cumsum()]
    weights = np.zeros((count, sizes.sum()))
    for b in range(count):
        for i in rng.integers(len(sizes), size=len(sizes)):
            picks = rng.integers(sizes[i], size=sizes[i])
            weights[b, offsets[i] : offsets[i + 1]] += np.bincount(picks, minlength=sizes[i]) / (len(sizes) * sizes[i])
    return weights


def paired_difference(left, right, *, confidence=0.95, identifiable=True):
    """Percentile interval of paired bootstrap differences; nonfinite samples stay invalid."""
    delta = np.asarray(left, float) - np.asarray(right, float)
    good = np.isfinite(delta)
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(delta[good], [tail, 1 - tail]) if good.any() else (np.nan, np.nan)
    return {
        "lower": float(lo) if identifiable else None,
        "upper": float(hi) if identifiable else None,
        "valid_fraction": float(good.mean()),
    }


def clustered_interval(clusters, weights, *, confidence=0.95):
    """Equal-initial point estimate and hierarchical percentile interval for scalars."""
    values = np.concatenate(clusters)
    center = float(np.mean([np.mean(c) for c in clusters]))
    tail = (1 - confidence) / 2
    boot = weights @ values
    lo, hi = np.quantile(boot, [tail, 1 - tail])
    return {
        "mean": center,
        "lower": float(lo) if len(clusters) > 1 else None,
        "upper": float(hi) if len(clusters) > 1 else None,
    }
