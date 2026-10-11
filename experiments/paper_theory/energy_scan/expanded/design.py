"""Triangular feasible-rate scan requests and scaled-time stopping protocols."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from experiments.paper_theory.energy_scan.expanded import config
from theory.runtime.kinetic import KineticParameters, KineticStopping, kinetic_request


def feasible_ratios(alpha: float) -> np.ndarray:
    """Return zero and declared positive ratios satisfying q=alpha*ratio <= 1, retaining their design order."""
    positive = config.POSITIVE_RATIOS[alpha * config.POSITIVE_RATIOS <= 1.0 + 1e-12]
    return np.concatenate(([0.0], positive))


def rewiring_rate(alpha: float, ratio: float) -> float:
    """Return alpha*ratio, snapping floating-point values within tolerance of one to exactly one."""
    value = alpha * ratio
    return 1.0 if math.isclose(value, 1.0, abs_tol=1e-12, rel_tol=0.0) else value


def step_controls(alpha: float) -> tuple[int, int, KineticStopping]:
    """Convert scaled horizon/recording/check/patience settings to physical steps and return (steps, record_every, stopping)."""
    steps = math.ceil(config.MAX_TAU / alpha)
    record_every = max(1, round(config.RECORD_TAU / alpha))
    stopping = KineticStopping(
        mode="state_and_energy",
        minimum_steps=math.ceil(config.MINIMUM_TAU / alpha),
        check_every=max(1, round(config.CHECK_TAU / alpha)),
        patience_steps=math.ceil(config.PATIENCE_TAU / alpha),
        state_l1_tolerance=config.STATE_RATE_TOLERANCE_TAU * alpha,
        energy_absolute_tolerance=config.ENERGY_ABSOLUTE_TOLERANCE_TAU * alpha,
        energy_relative_tolerance=config.ENERGY_RELATIVE_TOLERANCE_TAU * alpha,
    )
    return (steps, record_every, stopping)


def simulation_protocol(dynamics: str, alpha: float, ratios: np.ndarray, *, fingerprint=True) -> dict[str, Any]:
    """Return numerical scan provenance; fingerprint=False omits current binary/adapter hashes for read-only historical compatibility checks."""
    from experiments.paper_theory.energy_scan.expanded.storage import sha256

    steps, record_every, stopping = step_controls(alpha)
    return {
        "schema": 1,
        "definition": "expanded log-uniform alpha and q/alpha interaction-energy scan",
        "solver_binary_sha256": sha256(config.KINETIC) if fingerprint else None,
        "python_adapter_sha256": sha256(config.ADAPTER) if fingerprint else None,
        "dynamics": dynamics,
        "recommender": "random",
        "alpha": alpha,
        "epsilons": config.EPSILONS.tolist(),
        "q_over_alpha": ratios.tolist(),
        "q": [rewiring_rate(alpha, float(ratio)) for ratio in ratios],
        "grid_size": config.GRID_SIZE,
        "maximum_tau": config.MAX_TAU,
        "maximum_steps": steps,
        "record_tau_target": config.RECORD_TAU,
        "record_every": record_every,
        "stopping": asdict(stopping),
        "opinion_method": "measure",
        "mean_degree": 15,
        "recommendation_count": 10,
        "dt": 1.0,
        "noise_diffusion": 0.0,
        "confidence_mode": "cell_average",
    }


def model_parameters(
    dynamics: str, epsilon: float, alpha: float, ratio: float, *, steps: int, record_every: int
) -> KineticParameters:
    """Build one fixed-resolution Random-recommender parameter set with explicit q and requested step controls."""
    return KineticParameters(
        epsilon=float(epsilon),
        influence=float(alpha),
        rewiring=rewiring_rate(alpha, ratio),
        dynamics=dynamics,
        opinion_method="measure",
        mean_degree=15,
        recsys_count=10,
        recsys="random",
        opinion_tolerance=0.4,
        recommendation_steepness=1.0,
        recommendation_random_ratio=0.0,
        noise_diffusion=0.0,
        grid_size=config.GRID_SIZE,
        dt=1.0,
        steps=steps,
        record_every=record_every,
        confidence_mode="cell_average",
    )


def case_request(
    dynamics: str,
    alpha: float,
    epsilon: float,
    ratio: float,
    *,
    steps: int,
    record_every: int,
    stopping: KineticStopping,
) -> dict[str, object]:
    """Build a full energy trajectory request with joint stopping and final potential snapshots."""
    label = f"{dynamics}/a{alpha:.12g}/e{epsilon:g}/r{ratio:.12g}"
    return kinetic_request(
        label,
        model_parameters(dynamics, epsilon, alpha, ratio, steps=steps, record_every=record_every),
        config.RESOLUTION,
        observable_fields=config.ENERGY_NAMES,
        final_snapshot_fields=("node_potential", "edge_potential"),
        stopping=stopping,
    )


def probe_request(dynamics: str, alpha: float, epsilon: float, ratio: float) -> dict[str, object]:
    """Build a one-step energy request restoring first-step drops independently of coarse trajectory recording."""
    label = f"{dynamics}/a{alpha:.12g}/e{epsilon:g}/r{ratio:.12g}"
    return kinetic_request(
        f"first-step/{label}",
        model_parameters(dynamics, epsilon, alpha, ratio, steps=1, record_every=1),
        config.RESOLUTION,
        observable_fields=config.ENERGY_NAMES,
        stopping=KineticStopping(),
    )
