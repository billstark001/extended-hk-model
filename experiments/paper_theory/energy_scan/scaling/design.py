"""Isolated alpha/q probes and fixed-ratio dynamic-collapse protocols."""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.paper_theory.energy_scan.scaling import config
from theory.runtime.kinetic import KineticParameters


def base_parameters(*, alpha: float, q: float, steps: int, record_every: int) -> KineticParameters:
    """Build the common uniform-law Random model parameters used by channel-scaling probes."""
    return KineticParameters(
        epsilon=config.EPSILON,
        influence=alpha,
        rewiring=q,
        dynamics="hk",
        opinion_method="measure",
        mean_degree=config.MEAN_DEGREE,
        recsys_count=config.RECOMMENDATION_COUNT,
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


def protocol() -> dict[str, object]:
    """Record channel/dynamic probe grids and current solver/adapter identities for cache compatibility."""
    from common.runtime.git import revision

    return {
        "definition": "alpha/q energy-channel and scaled-time tests",
        "solver_revision": revision(config.GO_ROOT),
        "solver_binary_sha256": hashlib.sha256(config.KINETIC.read_bytes()).hexdigest(),
        "python_adapter_sha256": hashlib.sha256(Path(__file__).with_name("request.py").read_bytes()).hexdigest(),
        "grid_size": config.GRID_SIZE,
        "epsilon": config.EPSILON,
        "mean_degree": config.MEAN_DEGREE,
        "recommendation_count": config.RECOMMENDATION_COUNT,
        "dynamics": "hk",
        "opinion_method": "measure",
        "recommender": "random",
        "dt": 1.0,
        "alpha_probe": config.ALPHA_PROBE.tolist(),
        "q_probe": config.Q_PROBE.tolist(),
        "alpha_collapse": config.ALPHA_COLLAPSE.tolist(),
        "gamma_collapse": config.GAMMA_COLLAPSE.tolist(),
        "tau": config.TAU.tolist(),
        "epsilon_probe": config.EPSILON_PROBE.tolist(),
    }
