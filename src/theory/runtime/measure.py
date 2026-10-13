"""Request adapter for the public mesoscopic runtime."""

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from typing_extensions import Unpack

from common.types import PathInput
from theory.parameters import Parameters
from theory.types import FloatVector, MeasureOptions, MeasureResult, StoppingOptions


def measure_request(
    identifier: str,
    parameters: Parameters,
    *,
    initial_mass: FloatVector | None = None,
    final_state: bool = True,
    snapshot_taus: Sequence[float] = (),
    stopping: StoppingOptions | None = None,
) -> dict[str, Any]:
    """Translate Parameters into a public kinetic request with cell masses on [-1,1].

    Optional initial_mass must be a normalized nonnegative bins-vector. Convert
    scaled horizon/record_tau to physical steps, retaining initial/first-step
    energy snapshots; final_state controls final rho/edge fields.
    """
    p = parameters
    p.validate()
    if not identifier:
        raise ValueError("a request identifier is required")
    mass = np.full(p.bins, 1 / p.bins) if initial_mass is None else np.asarray(initial_mass, dtype=float)
    if (
        mass.shape != (p.bins,)
        or not np.isfinite(mass).all()
        or np.any(mass < 0)
        or not np.isclose(mass.sum(), 1, rtol=0, atol=1e-10)
    ):
        raise ValueError("initial mass must be a normalized nonnegative bins-vector")
    steps = math.ceil(p.horizon / (p.alpha * p.dt) - 1e-12)
    interval = max(1, round(p.record_tau / (p.alpha * p.dt)))
    request: dict[str, Any] = {
        "request_id": identifier,
        "population": p.population,
        "opinion_bins": p.bins,
        "out_degree": p.degree,
        "recommendation_count": p.recommendations,
        "steps": steps,
        "record_every": interval,
        "dt": p.dt,
        "noise_diffusion": 0.0,
        "confidence_mode": "cell_average",
        "dynamics": {
            "type": p.dynamics,
            "opinion_method": "measure",
            "tolerance": p.epsilon,
            "influence": p.alpha,
            "rewiring_rate": p.q,
        },
        "recommender": {
            "type": p.recommender,
            "steepness": p.steepness,
            "random_ratio": 0.0,
            "opinion_tolerance": p.opinion_tolerance,
        },
        "initial": {"type": "categorical", "opinion_min": -1.0, "opinion_max": 1.0, "probabilities": mass.tolist()},
        "resolution": {
            "opinion_quadrature_points": p.quadrature,
            "opinion_quadrature_rule": "gauss_hermite",
            "confidence_quadrature_points": 8,
            "score_max": 64,
            "distance_grid_size": 512,
        },
        "observables": {
            "polarization": True,
            "subjective": False,
            "homophily": True,
            "homophily_raw": False,
            "node_energy": True,
            "edge_energy": True,
            "pathway": False,
            "polarization_first_passage": False,
            "homophily_first_passage": False,
            "polarization_threshold": 0.8,
            "homophily_threshold": 0.8,
            "minimum_bandwidth": 0.02,
            "objective_effective_samples": 500,
        },
        "snapshots": {
            "record_steps": sorted({0, 1, *(min(steps, round(t / (p.alpha * p.dt))) for t in snapshot_taus)}),
            "rho": True,
            "edge": bool(snapshot_taus),
            "velocity": False,
            "rewiring_flux": False,
            "node_potential": True,
            "edge_potential": True,
            "final_rho": final_state,
            "final_edge": final_state,
            "final_node_potential": False,
            "final_edge_potential": False,
        },
        "stopping": {
            "mode": "fixed_steps",
            "minimum_steps": 0,
            "check_every": 1,
            "patience_steps": 1,
            "state_l1_tolerance": 0.0,
            "energy_absolute_tolerance": 0.0,
            "energy_relative_tolerance": 0.0,
        },
    }

    if stopping is not None:
        request["stopping"].update(stopping)
    return request


def run_measure(
    binary: PathInput, identifier: str, parameters: Parameters, **options: Unpack[MeasureOptions]
) -> MeasureResult:
    """Use the binding's exact codec with cancellable subprocess transport."""
    import json

    from smp_meso_bindings.kinetic_runner import decode_kinetic_response, prepare_kinetic_request

    from common.runtime.process import run_jsonl

    request = prepare_kinetic_request(measure_request(identifier, parameters, **options))
    response = decode_kinetic_response(json.loads(run_jsonl(binary, json.dumps(request) + "\n", arguments=("batch",))))
    if (
        response.get("request_id") != identifier
        or response.get("error")
        or not isinstance(response.get("result"), dict)
    ):
        raise RuntimeError(f"{identifier}: invalid kinetic response: {response.get('error')}")
    return cast(MeasureResult, response["result"])
