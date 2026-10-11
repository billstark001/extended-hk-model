"""Public-runtime generation and cache validation for channel and dynamic scaling."""

from __future__ import annotations

import json

import numpy as np

from experiments.paper_theory.energy_scan.scaling import config
from theory.runtime.kinetic import KineticParameters, KineticResolution, kinetic_request, run_kinetic


def request_snapshots(
    request_id: str, parameters: KineticParameters, record_steps: np.ndarray, fields: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Run a selected snapshot/observable request with the channel-scaling physical parameters."""
    request = kinetic_request(
        request_id, parameters, KineticResolution(), record_steps=record_steps, snapshot_fields=fields
    )
    response = run_kinetic(config.KINETIC, request)
    result = response.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("snapshots"), dict):
        raise TypeError(f"{request_id}: response contains no snapshots")
    snapshots = result["snapshots"]
    arrays = {field: np.asarray(snapshots[field], dtype=float) for field in fields}
    diagnostics_raw = result.get("diagnostics", {})
    diagnostics = {}
    if isinstance(diagnostics_raw, dict):
        for key in ("max_node_mass_residual", "max_fixed_degree_residual"):
            if key in diagnostics_raw:
                diagnostics[key] = float(diagnostics_raw[key])
    return (np.asarray(snapshots["time"], dtype=float), arrays, diagnostics)


def run_channel_probes() -> dict[str, np.ndarray | float]:
    """Measure isolated small-alpha and small-q first-step energy changes on the common initial law."""
    from experiments.paper_theory.energy_scan.scaling.design import base_parameters
    from theory.channel_scaling import (
        continuum_alpha_coefficient,
        continuum_q_coefficient,
        edge_energy,
        interaction_matrix,
        node_energy,
        opinion_grid,
    )

    x = opinion_grid()
    interaction = interaction_matrix(x)
    alpha_delta_u = np.empty_like(config.ALPHA_PROBE)
    for index, alpha in enumerate(config.ALPHA_PROBE):
        parameters = base_parameters(alpha=float(alpha), q=0.0, steps=1, record_every=1)
        _, snapshots, _ = request_snapshots(f"alpha-probe/{index}", parameters, np.asarray((0, 1)), ("rho",))
        values = node_energy(snapshots["rho"], interaction)
        alpha_delta_u[index] = values[1] - values[0]
    q_delta_ue = np.empty_like(config.Q_PROBE)
    q_delta_topology = np.empty_like(config.Q_PROBE)
    uniform_rho = np.full((2, config.GRID_SIZE), 1.0 / config.GRID_SIZE)
    uniform_u = node_energy(uniform_rho, interaction)
    for index, q in enumerate(config.Q_PROBE):
        parameters = base_parameters(alpha=0.0, q=float(q), steps=1, record_every=1)
        _, snapshots, _ = request_snapshots(f"q-probe/{index}", parameters, np.asarray((0, 1)), ("edge",))
        values = edge_energy(snapshots["edge"], interaction)
        q_delta_ue[index] = values[1] - values[0]
        topology = values - 2.0 * uniform_u
        q_delta_topology[index] = topology[1] - topology[0]
    return {
        "alpha_delta_u": alpha_delta_u,
        "q_delta_ue": q_delta_ue,
        "q_delta_topology": q_delta_topology,
        "alpha_coefficient_continuum": continuum_alpha_coefficient(),
        "q_coefficient_continuum": continuum_q_coefficient(),
    }


def run_dynamic_collapse() -> dict[str, np.ndarray]:
    """Generate energy and field curves across alpha at fixed q/alpha for scaled-time collapse checks."""
    from experiments.paper_theory.energy_scan.scaling.design import base_parameters
    from theory.channel_scaling import derived_fields, interaction_matrix, opinion_grid

    shape_scalar = (config.GAMMA_COLLAPSE.size, config.ALPHA_COLLAPSE.size, config.TAU.size)
    shape_field = shape_scalar + (config.GRID_SIZE,)
    energy_rho = np.empty(shape_scalar)
    energy_edge = np.empty(shape_scalar)
    topology_excess = np.empty(shape_scalar)
    phi_rho = np.empty(shape_field)
    phi_edge = np.empty(shape_field)
    max_node_residual = np.empty((config.GAMMA_COLLAPSE.size, config.ALPHA_COLLAPSE.size))
    max_degree_residual = np.empty_like(max_node_residual)
    x = opinion_grid()
    interaction = interaction_matrix(x)
    for gamma_index, gamma in enumerate(config.GAMMA_COLLAPSE):
        for alpha_index, alpha in enumerate(config.ALPHA_COLLAPSE):
            step_values = np.rint(config.TAU / alpha).astype(np.int64)
            if not np.allclose(alpha * step_values, config.TAU, atol=1e-13, rtol=0.0):
                raise ValueError("tau grid is incompatible with alpha grid")
            q = float(gamma * alpha)
            parameters = base_parameters(
                alpha=float(alpha), q=q, steps=int(step_values[-1]), record_every=int(step_values[1] - step_values[0])
            )
            print(f"collapse gamma={gamma:g} alpha={alpha:g} q={q:g} steps={parameters.steps}", flush=True)
            time, snapshots, diagnostics = request_snapshots(
                f"collapse/g{gamma_index}/a{alpha_index}", parameters, step_values, ("rho", "edge")
            )
            if not np.allclose(alpha * time, config.TAU, atol=1e-13, rtol=0.0):
                raise ValueError("solver times do not match requested scaled times")
            values = derived_fields(snapshots["rho"], snapshots["edge"], interaction)
            phi_rho[gamma_index, alpha_index] = values[0]
            phi_edge[gamma_index, alpha_index] = values[1]
            energy_rho[gamma_index, alpha_index] = values[2]
            energy_edge[gamma_index, alpha_index] = values[3]
            topology_excess[gamma_index, alpha_index] = values[4]
            max_node_residual[gamma_index, alpha_index] = diagnostics.get("max_node_mass_residual", np.nan)
            max_degree_residual[gamma_index, alpha_index] = diagnostics.get("max_fixed_degree_residual", np.nan)
    return {
        "x": x,
        "energy_rho": energy_rho,
        "energy_edge": energy_edge,
        "topology_excess": topology_excess,
        "phi_rho": phi_rho,
        "phi_edge": phi_edge,
        "max_node_residual": max_node_residual,
        "max_degree_residual": max_degree_residual,
    }


def uniform_epsilon_curves() -> dict[str, np.ndarray]:
    """Evaluate uniform-start analytic energy coefficients across the confidence grid."""
    from theory.channel_scaling import opinion_grid

    x = opinion_grid()
    rho = np.full(config.GRID_SIZE, 1.0 / config.GRID_SIZE)
    numerical_u = np.empty_like(config.EPSILON_PROBE)
    numerical_ue = np.empty_like(config.EPSILON_PROBE)
    numerical_du = np.empty_like(config.EPSILON_PROBE)
    numerical_due = np.empty_like(config.EPSILON_PROBE)
    for index, epsilon in enumerate(config.EPSILON_PROBE):
        distance = np.abs(x[:, None] - x[None, :])
        interaction = 0.5 * np.minimum(distance**2, epsilon**2)
        pair_mass = rho[:, None] * rho[None, :]
        numerical_u[index] = 0.5 * np.sum(interaction * pair_mass)
        numerical_ue[index] = 2.0 * numerical_u[index]
        discordant_probability = np.sum(pair_mass[distance > epsilon])
        numerical_du[index] = 0.5 * epsilon * discordant_probability
        numerical_due[index] = epsilon * discordant_probability
    epsilon = config.EPSILON_PROBE
    analytic_u = epsilon**2 / 4.0 - epsilon**3 / 6.0 + epsilon**4 / 32.0
    analytic_ue = 2.0 * analytic_u
    analytic_du = epsilon / 2.0 - epsilon**2 / 2.0 + epsilon**3 / 8.0
    analytic_due = 2.0 * analytic_du
    return {
        "epsilon_numerical_u": numerical_u,
        "epsilon_numerical_ue": numerical_ue,
        "epsilon_numerical_du": numerical_du,
        "epsilon_numerical_due": numerical_due,
        "epsilon_analytic_u": analytic_u,
        "epsilon_analytic_ue": analytic_ue,
        "epsilon_analytic_du": analytic_du,
        "epsilon_analytic_due": analytic_due,
    }


def run_all() -> dict[str, np.ndarray]:
    """Generate the complete channel-scaling dataset and its declared numerical diagnostics."""
    from experiments.paper_theory.energy_scan.scaling.design import protocol

    output: dict[str, np.ndarray] = {
        "protocol": np.asarray(json.dumps(protocol(), sort_keys=True)),
        "alpha_probe": config.ALPHA_PROBE,
        "q_probe": config.Q_PROBE,
        "alpha_collapse": config.ALPHA_COLLAPSE,
        "gamma_collapse": config.GAMMA_COLLAPSE,
        "tau": config.TAU,
        "epsilon_probe": config.EPSILON_PROBE,
    }
    output.update({key: np.asarray(value) for key, value in run_channel_probes().items()})
    output.update(run_dynamic_collapse())
    output.update(uniform_epsilon_curves())
    np.savez_compressed(config.HERE / "energy_parameter_scaling.npz", **output)
    return output


def load_or_run(*, analysis_only=False) -> dict[str, np.ndarray]:
    """Read a compatible channel-scaling cache or generate a fresh dataset; analysis_only never runs the solver."""
    from experiments.paper_theory.energy_scan.scaling.design import protocol

    cache = (config.INPUT if analysis_only else config.HERE) / "energy_parameter_scaling.npz"
    fingerprint = None if analysis_only else json.dumps(protocol(), sort_keys=True)
    if cache.is_file():
        with np.load(cache) as data:
            if not analysis_only and str(data["protocol"].item()) != fingerprint:
                raise ValueError(f"incompatible cache: {cache}")
            print("reusing compatible energy-scaling cache", flush=True)
            return {name: np.asarray(data[name]) for name in data.files}
    if analysis_only:
        raise FileNotFoundError(cache)
    return run_all()
