"""Discretize a declared microscopic initial law for pair-model requests."""

import numpy as np


def initial_mass(b, spec):
    """Return normalized midpoint masses for a uniform or Gaussian-mixture specification; microscopic clipping is separate."""
    x = spec["minimum"] + (np.arange(b) + 0.5) * (spec["maximum"] - spec["minimum"]) / b
    if spec["type"] == "uniform":
        p = np.ones(b)
    else:
        p = sum(c["weight"] / c["std"] * np.exp(-0.5 * ((x - c["mean"]) / c["std"]) ** 2) for c in spec["components"])
    return p / p.sum()
