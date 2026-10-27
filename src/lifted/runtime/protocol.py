"""Translate frozen tasks to public model requests; no experiment design defaults."""

from common.io.json import fingerprint
from lifted.types import Task
from theory.parameters import Parameters
from theory.runtime.measure import measure_request


def stream_seed(seed: int, *identity) -> int:
    """Scheduling-independent uint64 namespace for every continuation index."""
    return int(fingerprint({"seed": seed, "identity": identity})[:16], 16)


def model_request(task: Task):
    p = Parameters(**task["parameters"])
    p.validate()
    if p.dynamics != "hk" or p.recommender != "random" or p.dt != 1:
        raise ValueError("this workflow requires HK + Random and physical dt=1")
    return {
        "request_id": task["id"],
        "layer": task["layer"],
        "population": p.population,
        "opinion_bins": p.bins,
        "out_degree": p.degree,
        "recommendation_count": p.recommendations,
        "max_steps": max(1, task["horizon_steps"]),
        "paths": 1,
        "interval_paths": 1,
        "ambiguity_samples": 1,
        "confidence_level": 0.95,
        "workers": 1,
        "seed": task["seed"],
        "major_cluster_mass": 0.5 / p.population,
        "terminal_position_resolution": 0,
        "terminal_mass_resolution": 0,
        "dynamics": {"type": p.dynamics, "tolerance": p.epsilon, "influence": p.alpha, "rewiring_rate": p.q},
        "recommender": {
            "type": "random",
            "steepness": 1,
            "random_ratio": 0,
            "opinion_tolerance": 0.4,
            "noise_std": 0,
            "noise_quadrature_points": 9,
        },
        "initial": {"type": "uniform", "opinion_min": -1, "opinion_max": 1, "probabilities": []},
        "resolution": {
            "score_max": 64,
            "availability_bins": 2,
            "component_size_bins": 2,
            "opinion_quadrature_points": p.quadrature,
            "opinion_quadrature_rule": "gauss_hermite",
        },
        "closure": {
            "motif_relaxation": 0.25,
            "histogram_relaxation": 0.25,
            "candidate_relaxation": 0.25,
            "topology_relaxation": 0.25,
        },
        "fast_slow": {
            "mode": "unsplit",
            "ratio_threshold": 10,
            "max_substeps": 1,
            "zero_event_batches": 1,
            "residual_tolerance": 0,
            "zero_event_residual": 0,
        },
        "ambiguity": {
            "eligibility_correlation_radius": 0,
            "score_availability_radius": 0,
            "motif_persistence_radius": 0,
            "bridge_bias_radius": 0,
            "component_mix_radius": 0,
        },
    }


def request(task, source, directory, *, source_kind, mode="branch"):
    from smp_meso_bindings.kinetic_runner import prepare_kinetic_request

    p = Parameters(**task["parameters"])
    measure = measure_request(task["id"], p, final_state=False)
    measure["steps"] = max(1, task["horizon_steps"])
    measure["record_every"] = max(1, round(0.1 / p.alpha))
    measure["snapshots"]["record_steps"] = sorted(set([0, *task["observe_steps"]]))
    return dict(
        id=task["id"],
        kind=task["kind"],
        source=str(source["checkpoint"]),
        source_sha256=source["checkpoint_sha256"],
        source_kind=source_kind,
        mode=mode,
        steps=task["horizon_steps"],
        observe_steps=task["observe_steps"],
        seeds=[
            {"replicate": i, "value": stream_seed(task["seed"], task["source_task"], task["source_step"], i)}
            for i in task["replicates"]
        ],
        output_dir=str(directory),
        variant=task["variant"],
        model=model_request(task),
        measure=prepare_kinetic_request(measure),
        bridge_agents=0,
        group_center=task["group_center"],
    )
