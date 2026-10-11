"""Aggregate scan completeness, conservation and fitted-model quality reports."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from experiments.paper_theory.energy_scan.expanded import config


def build_summary(rows: list[dict[str, Any]], fits: list[dict[str, Any]]) -> dict[str, Any]:
    """Build and persist the expanded scan summary from raw shards and derived descriptor/fit tables."""
    summary: dict[str, Any] = {
        "design": {
            "dynamics": list(config.DYNAMICS),
            "recommender": "random",
            "epsilon": config.EPSILONS.tolist(),
            "alpha": config.ALPHAS.tolist(),
            "positive_q_over_alpha": config.POSITIVE_RATIOS.tolist(),
            "q_zero_separate": True,
            "main_trajectory_count": len(rows) // len(config.ENERGY_NAMES),
            "maximum_q_over_alpha": float(config.POSITIVE_RATIOS[-1]),
        },
        "quality": {
            "converged": sum((bool(row["converged"]) for row in rows[::2])),
            "maximum_node_mass_residual": max((float(row["maximum_node_mass_residual"]) for row in rows)),
            "maximum_fixed_degree_residual": max((float(row["maximum_fixed_degree_residual"]) for row in rows)),
            "minimum_nonincreasing_fraction": min((float(row["nonincreasing_fraction"]) for row in rows)),
            "maximum_executed_tau": max((float(row["executed_tau"]) for row in rows)),
        },
        "compression": {},
    }
    for dynamics in config.DYNAMICS:
        summary["compression"][dynamics] = {}
        for energy in config.ENERGY_NAMES:
            summary["compression"][dynamics][energy] = {}
            for model in sorted({row["model"] for row in fits}):
                subset = [
                    row
                    for row in fits
                    if row["dynamics"] == dynamics and row["energy"] == energy and (row["model"] == model)
                ]
                maximum_errors = np.asarray([row["maximum_absolute_relative_error"] for row in subset])
                summary["compression"][dynamics][energy][model] = {
                    "median_maximum_relative_error": float(np.median(maximum_errors)),
                    "p90_maximum_relative_error": float(np.quantile(maximum_errors, 0.9)),
                    "median_rms_log_time_error": float(np.median([row["rms_log_time_error"] for row in subset])),
                }
    (config.HERE / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
