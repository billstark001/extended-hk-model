"""Terminal energy, potential-well and derivative descriptors for scan curves."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.paper_theory.energy_scan.expanded import config
from theory.potentials import count_potential_wells


def terminal_state_parameters(
    shards: dict[tuple[str, float], dict[str, np.ndarray]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return endpoint energy/well descriptors across available shard cells using saved final potential arrays."""
    from experiments.paper_theory.energy_scan.expanded.design import rewiring_rate

    energy_rows = {
        (
            row["dynamics"],
            round(float(row["alpha"]), 12),
            round(float(row["epsilon"]), 12),
            round(float(row["q_over_alpha"]), 12),
            row["energy"],
        ): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for dynamics in config.DYNAMICS:
        for alpha_value in config.ALPHAS:
            alpha = float(alpha_value)
            shard = shards[dynamics, alpha]
            for epsilon_index, epsilon_value in enumerate(config.EPSILONS):
                epsilon = float(epsilon_value)
                for ratio_index, ratio_value in enumerate(shard["q_over_alpha"]):
                    ratio = float(ratio_value)

                    def energy_row(energy: str) -> dict[str, Any]:
                        return energy_rows[dynamics, round(alpha, 12), round(epsilon, 12), round(ratio, 12), energy]

                    node_row = energy_row("node_energy")
                    edge_row = energy_row("edge_energy")
                    output.append(
                        {
                            "dynamics": dynamics,
                            "epsilon": epsilon,
                            "alpha": alpha,
                            "q": rewiring_rate(alpha, ratio),
                            "q_over_alpha": ratio,
                            "node_final_energy": float(node_row["plateau"]),
                            "edge_final_energy": float(edge_row["plateau"]),
                            "node_final_energy_fraction": float(node_row["plateau"] / node_row["initial_energy"]),
                            "edge_final_energy_fraction": float(edge_row["plateau"] / edge_row["initial_energy"]),
                            "node_potential_well_count": count_potential_wells(
                                shard["final_node_potential"][epsilon_index, ratio_index]
                            ),
                            "well_prominence_fraction": config.WELL_PROMINENCE_FRACTION,
                            "trajectory_converged": bool(shard["converged"][epsilon_index, ratio_index]),
                        }
                    )
    return output
