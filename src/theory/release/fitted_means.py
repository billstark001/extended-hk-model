"""Infinite fitted clocks and explicitly unobserved survival tail fractions."""

import numpy as np
from scipy.special import gammaincc

from .means import gompertz_mean, gompertz_survival, mixture_mean, stretched_mean


def fitted_clocks(case, node_fit, edge_fit):
    """Return infinite empirical-fit means and unobserved tail fractions in tau units, retaining fit-quality flags."""
    h, g = float(node_fit["initial_hazard"]), float(node_fit["acceleration"])
    weight = float(edge_fit["weight"])
    mn = gompertz_mean(h, g)
    horizon = float(case["tau"][-1])
    if weight == 0:
        mf, me, tail_fast = np.nan, mn, 0.0
    else:
        rate, shape = float(edge_fit["kappa"]) * case["r"], float(edge_fit["beta"])
        mf = stretched_mean(rate, shape)
        me = mixture_mean(weight, mf, mn)
        tail_fast = mf * gammaincc(1 / shape, (rate * horizon) ** shape)
    gt = g * horizon
    tail_node = (float(gompertz_survival([horizon], h, g)[0]) * gompertz_mean(h * np.exp(gt), g)) if gt < 700 else 0.0
    return {
        "fitted_mean_node": mn,
        "fitted_mean_edge": me,
        "fitted_mean_fast": mf,
        "fitted_edge_to_node": me / mn,
        "weight": weight,
        "node_tail_fraction": tail_node / mn,
        "edge_tail_fraction": (weight * tail_fast + (1 - weight) * tail_node) / me,
        "fit_shape_identifiable": edge_fit["shape_identifiable"],
        "fit_low_quality": edge_fit["low_quality_fit"],
        "node_fit_max_quantile_error": float(node_fit["maximum_absolute_relative_error"]),
        "fit_edge_max_curve_error": float(edge_fit["maximum_absolute_normalized_energy_error"]),
    }
