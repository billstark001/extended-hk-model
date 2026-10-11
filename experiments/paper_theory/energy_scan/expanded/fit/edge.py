"""Profiled multistart fits of fast edge release plus the observed node survival."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.paper_theory.energy_scan.expanded import config
from theory.release.edge_fits import fit_edge_curve


def fit_edge_timescales(shards: dict[tuple[str, float], dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    """Fit S_E = w*exp(-(kappa*r*tau)**beta) + (1-w)*S_rho for each shard curve.

    Edge affine normalization and S_rho remain fixed from simulations. Fit sampled
    squared curve errors with multiple starts; report quality and identifiability.
    At r=0 set w=0 and leave kappa/beta NaN because the fast component is unidentifiable.
    """

    from experiments.paper_theory.energy_scan.expanded.design import rewiring_rate
    from theory.release.quantile_fits import survival_curve

    output: list[dict[str, Any]] = []
    for dynamics in config.DYNAMICS:
        for alpha_value in config.ALPHAS:
            alpha = float(alpha_value)
            shard = shards[dynamics, alpha]
            for epsilon_index, epsilon in enumerate(config.EPSILONS):
                for ratio_index, ratio_value in enumerate(shard["q_over_alpha"]):
                    ratio = float(ratio_value)
                    length = int(shard["lengths"][epsilon_index, ratio_index])
                    tau = shard["tau"][epsilon_index, ratio_index, :length]
                    node = shard["node_energy"][epsilon_index, ratio_index, :length]
                    edge = shard["edge_energy"][epsilon_index, ratio_index, :length]
                    if tau.size > 1 and tau[1] > alpha + 1e-12:
                        tau = np.insert(tau, 1, alpha)
                        node = np.insert(node, 1, node[0] + shard["first_step_delta"][epsilon_index, ratio_index, 0])
                        edge = np.insert(edge, 1, edge[0] + shard["first_step_delta"][epsilon_index, ratio_index, 1])
                    node_survival, node_plateau, node_amplitude = survival_curve(node)
                    edge_survival, edge_plateau, edge_amplitude = survival_curve(edge)
                    fitted = fit_edge_curve(tau, node_survival, edge_survival, ratio, config.EDGE_BETA_BOUNDS)
                    weight, kappa, beta = (fitted[k] for k in ("weight", "kappa", "beta"))
                    optimizer_success, jacobian_condition = (
                        fitted[k] for k in ("optimizer_success", "jacobian_condition")
                    )
                    error, rms_error = fitted["error"], fitted["rms_error"]
                    output.append(
                        {
                            "dynamics": dynamics,
                            "epsilon": float(epsilon),
                            "alpha": alpha,
                            "q": rewiring_rate(alpha, ratio),
                            "q_over_alpha": ratio,
                            "model": "rewiring_kww_plus_node_survival",
                            "edge_initial_energy": float(edge[0]),
                            "edge_plateau": edge_plateau,
                            "edge_drop_amplitude": edge_amplitude,
                            "node_plateau": node_plateau,
                            "node_drop_amplitude": node_amplitude,
                            "weight": weight,
                            "kappa": kappa,
                            "beta": beta,
                            "fast_rate_tau": kappa * ratio if ratio > 0 else float("nan"),
                            "fast_time_tau": 1.0 / (kappa * ratio) if ratio > 0 else float("nan"),
                            "jacobian_condition": jacobian_condition,
                            "rms_normalized_energy_error": rms_error,
                            "maximum_absolute_normalized_energy_error": float(np.max(np.abs(error))),
                            "optimizer_success": optimizer_success,
                            "low_quality_fit": bool(rms_error > 0.03),
                            "trajectory_converged": bool(shard["converged"][epsilon_index, ratio_index]),
                            "shape_identifiable": bool(
                                ratio > 0
                                and weight > 0.02
                                and (config.EDGE_BETA_BOUNDS[0] * 1.01 < beta)
                                and (beta < config.EDGE_BETA_BOUNDS[1] / 1.01)
                                and (jacobian_condition <= 1000.0)
                            ),
                        }
                    )
    return output
