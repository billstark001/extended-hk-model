"""Explicit request schema for archived energy scans; contains no solver.

Historical caches retain their recorded adapter hash. Newly generated caches
record this adapter and the current Go binary; they are a separate run.
"""

from dataclasses import asdict, dataclass

from theory.parameters import Parameters
from theory.runtime.measure import measure_request


@dataclass(frozen=True)
class KineticParameters:
    """Physical update probabilities and fixed-step controls for archived scan requests; rates are not implicitly scaled by influence."""

    epsilon: float = 0.4
    influence: float = 0.01
    rewiring: float = 0.02
    dynamics: str = "hk"
    opinion_method: str = "measure"
    mean_degree: int = 15
    recsys_count: int = 10
    recsys: str = "random"
    opinion_tolerance: float = 0.4
    recommendation_steepness: float = 1.0
    recommendation_random_ratio: float = 0.0
    noise_diffusion: float = 0.0
    grid_size: int = 161
    dt: float = 1.0
    steps: int = 10000
    record_every: int = 1
    confidence_mode: str = "cell_average"


@dataclass(frozen=True)
class KineticResolution:
    """Quadrature and numerical lookup resolutions passed directly to the public mesoscopic runtime."""

    opinion_quadrature_points: int = 9
    opinion_quadrature_rule: str = "gauss_hermite"
    confidence_quadrature_points: int = 8
    score_max: int = 64
    distance_grid_size: int = 512


@dataclass(frozen=True)
class KineticStopping:
    """Fixed-step or joint state/energy stopping thresholds and checkpoint/patience counts in runtime steps."""

    mode: str = "fixed_steps"
    minimum_steps: int = 0
    check_every: int = 1
    patience_steps: int = 1
    state_l1_tolerance: float = 0.0
    energy_absolute_tolerance: float = 0.0
    energy_relative_tolerance: float = 0.0


def kinetic_request(
    identifier,
    parameters,
    resolution,
    *,
    observable_fields=(),
    final_snapshot_fields=(),
    stopping=None,
    record_steps=(),
    snapshot_fields=(),
):
    """Build a public kinetic request for a continuous uniform initial opinion law.

    observable_fields and snapshot_fields opt into outputs; final_snapshot_fields
    contains names without the final_ prefix. Zero influence is allowed for
    rewiring-only probes. No solver import or execution occurs here.
    """
    p = parameters
    # The first-step splitting audit permits zero influence. Build the public
    # schema using a valid placeholder, then supply the physical update rates.
    request = measure_request(
        identifier,
        Parameters(
            dynamics=p.dynamics,
            recommender=p.recsys,
            epsilon=p.epsilon,
            alpha=0.01,
            ratio=0.0,
            degree=p.mean_degree,
            recommendations=p.recsys_count,
            bins=p.grid_size,
            dt=p.dt,
            horizon=0.01 * p.dt,
            steepness=p.recommendation_steepness,
            opinion_tolerance=p.opinion_tolerance,
        ),
    )
    request.update(
        steps=p.steps, record_every=p.record_every, noise_diffusion=p.noise_diffusion, confidence_mode=p.confidence_mode
    )
    request["dynamics"].update(influence=p.influence, rewiring_rate=p.rewiring, opinion_method=p.opinion_method)
    request["recommender"]["random_ratio"] = p.recommendation_random_ratio
    request["resolution"] = asdict(resolution)
    # Preserve the original continuous uniform initial law.
    request["initial"].update(type="uniform", probabilities=[])
    for key, value in request["observables"].items():
        if isinstance(value, bool):
            request["observables"][key] = key in observable_fields
    request["snapshots"]["record_steps"] = list(record_steps)
    for key, value in request["snapshots"].items():
        if isinstance(value, bool):
            request["snapshots"][key] = (
                key[6:] in final_snapshot_fields if key.startswith("final_") else key in snapshot_fields
            )
    request["stopping"] = asdict(stopping or KineticStopping())
    return request


def run_kinetic_batch_parallel(*args, **kwargs):
    """Lazily call the public parallel-batch binding, forwarding arguments and response order unchanged."""
    from smp_meso_bindings import run_kinetic_batch_parallel as run

    return run(*args, **kwargs)


def run_kinetic(*args, **kwargs):
    """Lazily call the public single-request binding, forwarding its arguments and response unchanged."""
    from smp_meso_bindings import run_kinetic as run

    return run(*args, **kwargs)
