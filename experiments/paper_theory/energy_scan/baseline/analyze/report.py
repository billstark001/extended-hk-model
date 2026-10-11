"""Baseline physical/numerical diagnostics and linear channel-scaling summaries."""

from __future__ import annotations

import json

import numpy as np

from experiments.paper_theory.energy_scan.baseline import config


def linear_fit_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return linear regression coefficients and goodness-of-fit diagnostics for channel scaling data."""
    positive = x > 0
    slope = float(np.dot(x[positive], y[positive]) / np.dot(x[positive], x[positive]))
    fitted = slope * x[positive]
    residual = float(np.sum((y[positive] - fitted) ** 2))
    total = float(np.sum((y[positive] - np.mean(y[positive])) ** 2))
    return {"slope": slope, "r_squared": 1.0 - residual / total}


def build_summary(
    data: dict[str, np.ndarray], rows: list[dict[str, object]], audit: dict[str, np.ndarray], recorded_protocol=None
) -> dict[str, object]:
    """Aggregate baseline conservation, convergence and channel audit results with their recorded protocol."""
    from experiments.paper_theory.energy_scan.baseline.design import protocol

    first_step: dict[str, object] = {}
    for index, epsilon in enumerate(config.EPSILONS):
        rewiring_drop = -audit["first_step_rewiring_only_delta"][index, :, 1]
        first_step[f"epsilon_{epsilon:g}"] = {
            "rewiring_only_drop_vs_q": linear_fit_summary(config.ALPHA * config.RATIOS, rewiring_drop),
            "maximum_abs_node_q_dependence": float(
                np.max(
                    np.abs(audit["first_step_full_delta"][index, :, 0] - audit["first_step_full_delta"][index, 0, 0])
                )
            ),
            "maximum_abs_splitting_residual": float(np.max(np.abs(audit["first_step_splitting_residual"][index]))),
            "q_1_full_edge_drop": float(-audit["first_step_full_delta"][index, -1, 1]),
            "q_1_rewiring_edge_drop": float(rewiring_drop[-1]),
        }
    trends: dict[str, object] = {}
    for epsilon in config.EPSILONS:
        for energy in config.ENERGY_NAMES:
            selected = [row for row in rows if row["epsilon"] == float(epsilon) and row["energy"] == energy]
            positive = selected[1:]
            trends[f"epsilon_{epsilon:g}_{energy}"] = {
                "tau_50_positive_range": [
                    float(np.nanmin([row["tau_50"] for row in positive])),
                    float(np.nanmax([row["tau_50"] for row in positive])),
                ],
                "tau_90_positive_range": [
                    float(np.nanmin([row["tau_90"] for row in positive])),
                    float(np.nanmax([row["tau_90"] for row in positive])),
                ],
                "plateau_positive_range": [
                    float(np.nanmin([row["plateau"] for row in positive])),
                    float(np.nanmax([row["plateau"] for row in positive])),
                ],
                "first_step_fraction_positive_range": [
                    float(np.nanmin([row["first_step_fraction"] for row in positive])),
                    float(np.nanmax([row["first_step_fraction"] for row in positive])),
                ],
            }
    summary = {
        "protocol": recorded_protocol if recorded_protocol is not None else protocol(),
        "scan": {
            "points": int(config.EPSILONS.size * config.RATIOS.size),
            "converged": int(np.count_nonzero(data["converged"])),
            "maximum_executed_steps": int(np.max(data["executed_steps"])),
            "maximum_node_mass_residual": float(np.max(data["maximum_node_mass_residual"])),
            "maximum_fixed_degree_residual": float(np.max(data["maximum_fixed_degree_residual"])),
        },
        "first_step_audit": first_step,
        "descriptor_trends": trends,
        "parameterization": {
            "curve": "U(t)=plateau+drop_amplitude*(1-progress(t))",
            "inverse_progress": "tau=T(p), p in progress_levels, tau=alpha*t",
            "shape": "(T(p)-T(0.1))/(T(0.9)-T(0.1))",
            "q_interpolation": "shape-preserving cubic PCHIP in log(1+q/alpha), positive q only",
            "q_zero": "stored separately because q->0+ and t->infinity need not commute",
        },
    }
    (config.HERE / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
