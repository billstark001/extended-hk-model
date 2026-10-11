"""Baseline uniform-law energy scan requests and provenance."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from experiments.paper_theory.energy_scan.baseline import config
from theory.runtime.kinetic import KineticParameters, kinetic_request


def protocol() -> dict[str, object]:
    """Return the baseline physical/numerical design with current solver and adapter provenance."""
    from common.runtime.git import git_fingerprint

    return {
        "definition": "dense fixed-alpha q/alpha interaction-energy curve parameterization",
        "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "solver": git_fingerprint(config.GO_ROOT),
        "python_adapter_sha256": hashlib.sha256(config.ADAPTER.read_bytes()).hexdigest(),
        "alpha": config.ALPHA,
        "epsilons": config.EPSILONS.tolist(),
        "q_over_alpha": config.RATIOS.tolist(),
        "q": (config.ALPHA * config.RATIOS).tolist(),
        "progress_levels": config.PROGRESS_LEVELS.tolist(),
        "grid_size": config.GRID_SIZE,
        "maximum_steps": config.MAX_STEPS,
        "record_every": config.RECORD_EVERY,
        "stopping": asdict(config.STOPPING),
        "dynamics": "hk",
        "opinion_method": "measure",
        "recommender": "random",
        "mean_degree": 15,
        "recommendation_count": 10,
        "dt": 1.0,
        "noise_diffusion": 0.0,
        "confidence_mode": "cell_average",
    }


def parameters(
    epsilon: float, ratio: float, *, alpha: float = config.ALPHA, steps: int = config.MAX_STEPS
) -> KineticParameters:
    """Build one baseline KineticParameters object for a confidence/rate-ratio cell."""
    return KineticParameters(
        epsilon=float(epsilon),
        influence=float(alpha),
        rewiring=float(config.ALPHA * ratio),
        dynamics="hk",
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
        record_every=config.RECORD_EVERY,
        confidence_mode="cell_average",
    )


def scan_request(epsilon: float, ratio: float) -> dict[str, object]:
    """Build a baseline energy request with the configured stopping, observable and final-snapshot choices."""
    return kinetic_request(
        f"energy-curve/e{epsilon:g}/r{ratio:.12g}",
        parameters(epsilon, ratio),
        config.RESOLUTION,
        observable_fields=config.ENERGY_NAMES,
        final_snapshot_fields=("node_potential", "edge_potential"),
        stopping=config.STOPPING,
    )
