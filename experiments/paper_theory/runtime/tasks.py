"""Public runtime requests, matched RNG seeds and frozen audit-task identities."""

from copy import deepcopy

from common.io.codec import encoded
from common.io.json import fingerprint
from theory.initial import initial_mass

_INITIAL_LAWS = {
    "uniform": {
        "type": "uniform",
        "minimum": -1.0,
        "maximum": 1.0,
        "clip_minimum": -0.999,
        "clip_maximum": 0.999,
        "components": [],
    },
    "bimodal": {
        "type": "gaussian_mixture",
        "minimum": -1.0,
        "maximum": 1.0,
        "clip_minimum": -0.999,
        "clip_maximum": 0.999,
        "components": [{"weight": 0.5, "mean": 0.55, "std": 0.12}, {"weight": 0.5, "mean": -0.55, "std": 0.12}],
    },
    "asymmetric": {
        "type": "gaussian_mixture",
        "minimum": -1.0,
        "maximum": 1.0,
        "clip_minimum": -0.999,
        "clip_maximum": 0.999,
        "components": [{"weight": 0.7, "mean": -0.35, "std": 0.15}, {"weight": 0.3, "mean": 0.65, "std": 0.10}],
    },
    "short_multimodal": {
        "type": "gaussian_mixture",
        "minimum": -1.0,
        "maximum": 1.0,
        "clip_minimum": -0.999,
        "clip_maximum": 0.999,
        "components": [
            {"weight": 1 / 1.3, "mean": -0.16, "std": 0.03},
            {"weight": 0.3 / 1.3, "mean": 0.22, "std": 0.03},
        ],
    },
}


def request(
    dynamics="hk",
    rec="random",
    zeta=1,
    alpha=0.02,
    ratio=2,
    epsilon=0.45,
    b=81,
    horizon=40,
    law="uniform",
    method="measure",
    h=1,
    adaptive=False,
    n=500,
    degree=15,
    recommendations=10,
    quadrature=9,
    confidence="cell_average",
    pathway=False,
):
    """Construct the public kinetic request used by audit designs, without invoking a solver.

    horizon uses tau=alpha*t and h is the physical step; ratio sets q/alpha.
    The temporary request_id remains a fixed placeholder until task identification.
    Encoded initial masses approximate the declared law on b uniform opinion cells.
    """
    q = alpha * ratio
    if not 0 <= q <= 1:
        raise ValueError("q outside [0,1]")
    steps = max(1, round(horizon / (alpha * h)))
    interval = max(1, round(0.1 / (alpha * h)))
    stopping = {
        "mode": "fixed_steps",
        "minimum_steps": 0,
        "check_every": 1,
        "patience_steps": 1,
        "state_l1_tolerance": 0.0,
        "energy_absolute_tolerance": 0.0,
        "energy_relative_tolerance": 0.0,
    }
    if adaptive:
        stopping.update(
            mode="state_and_energy",
            minimum_steps=min(steps, round(5 / (alpha * h))),
            check_every=min(steps, max(1, round(0.2 / (alpha * h)))),
            patience_steps=max(1, round(2 / (alpha * h))),
            state_l1_tolerance=1e-5 * alpha * h,
            energy_absolute_tolerance=1e-7 * alpha * h,
            energy_relative_tolerance=1e-5 * alpha * h,
        )
    return {
        "request_id": "revision",
        "population": n,
        "opinion_bins": b,
        "out_degree": degree,
        "recommendation_count": recommendations,
        "steps": steps,
        "record_every": interval,
        "dt": h,
        "noise_diffusion": 0.0,
        "confidence_mode": confidence,
        "dynamics": {
            "type": dynamics,
            "opinion_method": method,
            "tolerance": epsilon,
            "influence": alpha,
            "rewiring_rate": q,
        },
        "recommender": {"type": rec, "steepness": float(zeta), "random_ratio": 0.0, "opinion_tolerance": 0.4},
        "initial": {
            "type": "categorical",
            "opinion_min": _INITIAL_LAWS[law]["minimum"],
            "opinion_max": _INITIAL_LAWS[law]["maximum"],
            "probabilities": encoded(initial_mass(b, _INITIAL_LAWS[law])),
        },
        "resolution": {
            "opinion_quadrature_points": quadrature,
            "opinion_quadrature_rule": "gauss_hermite",
            "confidence_quadrature_points": 8,
            "score_max": 64,
            "distance_grid_size": 512,
        },
        "observables": {
            "polarization": pathway,
            "subjective": False,
            "homophily": pathway,
            "homophily_raw": False,
            "node_energy": True,
            "edge_energy": True,
            "pathway": pathway,
            "polarization_first_passage": False,
            "homophily_first_passage": False,
            "polarization_threshold": 0.8,
            "homophily_threshold": 0.8,
            "minimum_bandwidth": 0.02,
            "objective_effective_samples": 500,
        },
        "snapshots": {
            "record_steps": encoded([]),
            **{
                k: False
                for k in (
                    "rho",
                    "edge",
                    "velocity",
                    "rewiring_flux",
                    "node_potential",
                    "edge_potential",
                    "final_rho",
                    "final_edge",
                    "final_node_potential",
                    "final_edge_potential",
                )
            },
        },
        "stopping": stopping,
    }


def task(
    family,
    r,
    *,
    kind="trace",
    tier="core",
    law="uniform",
    network="fixed",
    replicate=0,
    branches=0,
    budgets=True,
    physical_dt=1,
    snapshots=True,
    selection="weighted",
    meta=None,
):
    """Copy a request into an identified study task with metadata and matched seeds.

    Initial seeds exclude recommender/rates so design cells share physical starts.
    Microscopic tasks include explicit clipping laws and conditional branch seeds;
    budgets/snapshots select recorded evidence, not new model stepping code.
    """
    r = deepcopy(r)
    alpha = r["dynamics"]["influence"]
    h = r["dt"]
    last = r["steps"]
    snap = (
        sorted({0, last, *[min(last, round(t / (alpha * h))) for t in (0.1, 0.5, 1, 2, 4, 8, 16, 32)]})
        if snapshots
        else []
    )
    # Seeds deliberately exclude recommender/rates: same physical starts across cells.
    seed = 202610060000 + replicate * 1009 + (0 if law == "uniform" else 1000000 if law == "bimodal" else 2000000)
    payload = {
        "kind": kind,
        "request": r,
        "physical_dt": physical_dt,
        "snapshot_steps": snap,
        "budgets": budgets,
        "initial": law,
        "network": network,
        "replicate": replicate,
        "init_seed": seed,
        "process_seed": seed + 50000000,
        "branches": branches,
        "audit_steps": sorted({0, min(last, round(1 / alpha)), min(last, round(4 / alpha))}) if branches else [],
        "selection": selection,
    }
    if kind == "micro":
        payload["microscopic_initial"] = deepcopy(_INITIAL_LAWS[law])
        payload["micro_options"] = {"post_retain_count": 1, "repost_rate": 0.0, "recommendation_noise_std": 0.0}
        payload["branch_seeds"] = {
            str(step): [payload["process_seed"] + step * 10000019 + (branch + 1) * 1009 for branch in range(branches)]
            for step in payload["audit_steps"]
        }
    if family == "kernel" and kind == "frozen" and (meta or {}).get("scan") == "hk_law":
        meta = {**meta, "sample_seed": 20261006, "source_locations": [-0.65, 0.0, 0.65]}
    md = {
        "family": family,
        "tier": tier,
        "law": law,
        "network": network,
        "replicate": replicate,
        "dynamics": r["dynamics"]["type"],
        "method": r["dynamics"]["opinion_method"],
        "recommender": r["recommender"]["type"],
        "zeta": r["recommender"]["steepness"],
        "alpha": alpha,
        "q": r["dynamics"]["rewiring_rate"],
        "ratio": r["dynamics"]["rewiring_rate"] / alpha,
        "epsilon": r["dynamics"]["tolerance"],
        "b": r["opinion_bins"],
        "n": r["population"],
        "h": h,
        "physical_dt": physical_dt,
        "selection": selection,
        **(meta or {}),
    }
    identity = fingerprint({"payload": payload, "meta": md})[:20]
    payload["id"] = identity
    payload["request"]["request_id"] = identity
    return {"id": identity, "family": family, "tier": tier, "meta": md, "payload": payload}


def identify(job):
    """Recompute and assign all job/request IDs in place after parameter edits, excluding the previous ID via the fixed placeholder."""
    payload = job["payload"]
    payload.pop("id", None)
    payload["request"]["request_id"] = "revision"
    identity = fingerprint({"payload": payload, "meta": job["meta"]})[:20]
    job["id"] = payload["id"] = payload["request"]["request_id"] = identity
    return job


RECS = (
    ("random", 1),
    ("opinion_random", 1),
    ("opinion_random", 4),
    ("structure_random_l0", 1),
    ("structure_random_l0", 4),
)
DYNAMICS = ("hk", "deffuant")
