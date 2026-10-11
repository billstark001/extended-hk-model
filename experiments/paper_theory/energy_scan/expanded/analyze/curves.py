"""Tail-normalized finite-window energy descriptors on the expanded scaled-time grid."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.paper_theory.energy_scan.expanded import config
from theory.release.descriptors import curve_descriptors


def parameterize(shards: dict[tuple[str, float], dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    """Reduce the complete configured shard grid to physical-coordinate descriptor rows."""
    from experiments.paper_theory.energy_scan.expanded.design import rewiring_rate

    rows = []
    for dynamics in config.DYNAMICS:
        for alpha_value in config.ALPHAS:
            alpha = float(alpha_value)
            shard = shards[dynamics, alpha]
            ratios = shard["q_over_alpha"]
            for epsilon_index, epsilon in enumerate(config.EPSILONS):
                for ratio_index, ratio in enumerate(ratios):
                    length = int(shard["lengths"][epsilon_index, ratio_index])
                    tau = shard["tau"][epsilon_index, ratio_index, :length]
                    for energy_index, energy in enumerate(config.ENERGY_NAMES):
                        values = shard[energy][epsilon_index, ratio_index, :length]
                        first_step_delta = shard["first_step_delta"][epsilon_index, ratio_index, energy_index]
                        if tau.size > 1 and tau[1] > alpha + 1e-12:
                            descriptor_tau = np.insert(tau, 1, alpha)
                            descriptor_values = np.insert(values, 1, values[0] + first_step_delta)
                        else:
                            descriptor_tau = tau
                            descriptor_values = values
                        descriptors = curve_descriptors(
                            descriptor_tau, descriptor_values, first_step_delta, alpha, config.PROGRESS_LEVELS
                        )
                        rows.append(
                            {
                                "dynamics": dynamics,
                                "recommender": "random",
                                "epsilon": float(epsilon),
                                "alpha": alpha,
                                "q": rewiring_rate(alpha, float(ratio)),
                                "q_over_alpha": float(ratio),
                                "energy": energy,
                                "executed_steps": int(shard["executed_steps"][epsilon_index, ratio_index]),
                                "executed_tau": alpha * int(shard["executed_steps"][epsilon_index, ratio_index]),
                                "converged": bool(shard["converged"][epsilon_index, ratio_index]),
                                "maximum_node_mass_residual": float(
                                    shard["maximum_node_mass_residual"][epsilon_index, ratio_index]
                                ),
                                "maximum_fixed_degree_residual": float(
                                    shard["maximum_fixed_degree_residual"][epsilon_index, ratio_index]
                                ),
                                **descriptors,
                            }
                        )
    return rows
