"""Baseline trajectory generation, first-step probes and read-only cache validation."""

from __future__ import annotations

import json

import numpy as np

from experiments.paper_theory.energy_scan.baseline import config
from theory.runtime.kinetic import KineticStopping, kinetic_request


def run_scan() -> dict[str, np.ndarray]:
    """Execute the complete baseline grid and pack variable-length energy trajectories and final potentials."""
    from smp_meso_bindings import run_kinetic_batch_parallel

    from experiments.paper_theory.energy_scan.baseline.design import scan_request

    requests = [scan_request(epsilon, ratio) for epsilon in config.EPSILONS for ratio in config.RATIOS]
    responses = run_kinetic_batch_parallel(config.KINETIC, requests, config.WORKERS)
    shape = (config.EPSILONS.size, config.RATIOS.size)
    lengths = np.empty(shape, dtype=np.int64)
    executed = np.empty(shape, dtype=np.int64)
    converged = np.empty(shape, dtype=bool)
    final_state_rate = np.empty(shape)
    final_node_rate = np.empty(shape)
    final_edge_rate = np.empty(shape)
    maximum_node_residual = np.empty(shape)
    maximum_degree_residual = np.empty(shape)
    final_node_potential = np.empty(shape + (config.GRID_SIZE,))
    final_edge_potential = np.empty_like(final_node_potential)
    unpacked: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for response in responses:
        result = response["result"]
        series = result["series"]
        unpacked.append(
            (
                np.asarray(series["time"], dtype=float),
                np.asarray(series["node_energy"], dtype=float),
                np.asarray(series["edge_energy"], dtype=float),
            )
        )
    maximum = max((item[0].size for item in unpacked))
    time = np.full(shape + (maximum,), np.nan)
    node_energy = np.full_like(time, np.nan)
    edge_energy = np.full_like(time, np.nan)
    for flat, response in enumerate(responses):
        epsilon_index, ratio_index = np.unravel_index(flat, shape)
        values = unpacked[flat]
        length = values[0].size
        lengths[epsilon_index, ratio_index] = length
        time[epsilon_index, ratio_index, :length] = values[0]
        node_energy[epsilon_index, ratio_index, :length] = values[1]
        edge_energy[epsilon_index, ratio_index, :length] = values[2]
        result = response["result"]
        diagnostics = result["diagnostics"]
        snapshots = result["snapshots"]
        executed[epsilon_index, ratio_index] = int(diagnostics["executed_steps"])
        converged[epsilon_index, ratio_index] = bool(diagnostics["converged"])
        final_state_rate[epsilon_index, ratio_index] = float(diagnostics["final_state_l1_rate"])
        final_node_rate[epsilon_index, ratio_index] = float(diagnostics["final_node_energy_rate"])
        final_edge_rate[epsilon_index, ratio_index] = float(diagnostics["final_edge_energy_rate"])
        maximum_node_residual[epsilon_index, ratio_index] = float(diagnostics["max_node_mass_residual"])
        maximum_degree_residual[epsilon_index, ratio_index] = float(diagnostics["max_fixed_degree_residual"])
        final_node_potential[epsilon_index, ratio_index] = snapshots["final_node_potential"]
        final_edge_potential[epsilon_index, ratio_index] = snapshots["final_edge_potential"]
    return {
        "lengths": lengths,
        "time": time,
        "node_energy": node_energy,
        "edge_energy": edge_energy,
        "executed_steps": executed,
        "converged": converged,
        "final_state_l1_rate": final_state_rate,
        "final_node_energy_rate": final_node_rate,
        "final_edge_energy_rate": final_edge_rate,
        "maximum_node_mass_residual": maximum_node_residual,
        "maximum_fixed_degree_residual": maximum_degree_residual,
        "final_node_potential": final_node_potential,
        "final_edge_potential": final_edge_potential,
    }


def run_first_step_audit(full: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Generate isolated first-step channel probes used to separate motion, rewiring and mixed effects."""
    from smp_meso_bindings import run_kinetic_batch_parallel

    from experiments.paper_theory.energy_scan.baseline.design import parameters

    requests = []
    fixed = KineticStopping()
    for epsilon in config.EPSILONS:
        for ratio in config.RATIOS:
            probe = parameters(epsilon, ratio, alpha=0.0, steps=1)
            requests.append(
                kinetic_request(
                    f"rewiring-step/e{epsilon:g}/r{ratio:.12g}",
                    probe,
                    config.RESOLUTION,
                    observable_fields=config.ENERGY_NAMES,
                    stopping=fixed,
                )
            )
    responses = run_kinetic_batch_parallel(config.KINETIC, requests, config.WORKERS)
    shape = (config.EPSILONS.size, config.RATIOS.size, 2)
    rewiring = np.empty(shape)
    for flat, response in enumerate(responses):
        epsilon_index, ratio_index = np.unravel_index(flat, shape[:2])
        series = response["result"]["series"]
        rewiring[epsilon_index, ratio_index, 0] = np.diff(series["node_energy"])[0]
        rewiring[epsilon_index, ratio_index, 1] = np.diff(series["edge_energy"])[0]
    full_delta = np.stack(
        (
            full["node_energy"][:, :, 1] - full["node_energy"][:, :, 0],
            full["edge_energy"][:, :, 1] - full["edge_energy"][:, :, 0],
        ),
        axis=2,
    )
    zero = int(np.flatnonzero(config.RATIOS == 0)[0])
    opinion = np.repeat(full_delta[:, zero : zero + 1, :], config.RATIOS.size, axis=1)
    splitting = full_delta - opinion - rewiring
    return {
        "first_step_full_delta": full_delta,
        "first_step_opinion_only_delta": opinion,
        "first_step_rewiring_only_delta": rewiring,
        "first_step_splitting_residual": splitting,
    }


def load_or_run(*, analysis_only=False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return cached or newly generated trajectory/probe arrays; analysis_only validates numerical compatibility without regeneration."""
    from experiments.paper_theory.energy_scan.baseline.design import protocol
    from experiments.paper_theory.energy_scan.baseline.storage import save_data

    path = (config.INPUT if analysis_only else config.HERE) / "energy_curve_parameterization.npz"
    fingerprint = None if analysis_only else json.dumps(protocol(), sort_keys=True)
    if path.is_file():
        with np.load(path) as archive:
            if not analysis_only and str(archive["protocol"].item()) != fingerprint:
                raise ValueError(f"incompatible cache: {path}")
            names = set(archive.files)
            audit_names = {name for name in names if name.startswith("first_step_")}
            ignored = {"protocol", "epsilon", "q_over_alpha", "q", "alpha", "progress_levels"}
            data = {name: np.asarray(archive[name]) for name in names - audit_names - ignored}
            audit = {name: np.asarray(archive[name]) for name in audit_names}
            return (data, audit)
    if analysis_only:
        raise FileNotFoundError(path)
    data = run_scan()
    audit = run_first_step_audit(data)
    save_data(data, audit)
    return (data, audit)
