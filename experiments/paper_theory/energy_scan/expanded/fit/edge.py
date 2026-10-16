"""Profiled multistart edge fits using the observed node survival in each slot."""

import numpy as np

from theory.release.edge_fits import fit_edge_curve
from theory.release.quantile_fits import survival_curve

from .. import config
from ..storage import curve_slots


def fit_slot(slot):
    """Keep each channel's observed affine reference fixed during edge fitting."""
    node, node_plateau, node_amplitude = survival_curve(slot["node_energy"])
    edge, edge_plateau, edge_amplitude = survival_curve(slot["edge_energy"])
    if not np.all(np.isfinite(node)) or not np.all(np.isfinite(edge)):
        return None  # A constant channel has no identifiable normalized release curve.
    ratio = slot["q_over_alpha"]
    fitted = fit_edge_curve(slot["tau"], node, edge, ratio, config.EDGE_BETA_BOUNDS)
    weight, kappa, beta = (fitted[k] for k in ("weight", "kappa", "beta"))
    return {
        **{k: slot[k] for k in ("dynamics", "epsilon", "alpha", "q", "q_over_alpha")},
        "model": "rewiring_kww_plus_node_survival",
        "edge_initial_energy": float(slot["edge_energy"][0]),
        "edge_plateau": edge_plateau,
        "edge_drop_amplitude": edge_amplitude,
        "node_plateau": node_plateau,
        "node_drop_amplitude": node_amplitude,
        "weight": weight,
        "kappa": kappa,
        "beta": beta,
        "fast_rate_tau": kappa * ratio if ratio > 0 else float("nan"),
        "fast_time_tau": 1.0 / (kappa * ratio) if ratio > 0 else float("nan"),
        "jacobian_condition": fitted["jacobian_condition"],
        "rms_normalized_energy_error": fitted["rms_error"],
        "maximum_absolute_normalized_energy_error": float(np.max(np.abs(fitted["error"]))),
        "optimizer_success": fitted["optimizer_success"],
        "low_quality_fit": bool(fitted["rms_error"] > 0.03),
        "trajectory_converged": slot["converged"],
        "shape_identifiable": bool(
            ratio > 0
            and weight > 0.02
            and config.EDGE_BETA_BOUNDS[0] * 1.01 < beta < config.EDGE_BETA_BOUNDS[1] / 1.01
            and fitted["jacobian_condition"] <= 1000.0
        ),
    }


def fit_edge_timescales(shards):
    """Fit the supplied slots, including a partial shard or a single slot."""
    return [row for slot in curve_slots(shards) if (row := fit_slot(slot)) is not None]
