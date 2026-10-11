"""Predeclared designs; all inputs are parameters and independent random seeds."""

from dataclasses import asdict, replace
from itertools import product

from common.io.json import fingerprint
from theory.parameters import Parameters

PROTOCOL = {
    "schema": 1,
    "bootstrap_seed": 87231,
    "bootstrap_samples": 1000,
    "confidence_level": 0.95,
    "plateau_window_tau": 20.0,
    "minimum_plateau_tau": 120.0,
    "plateau_relative_tolerance": 1e-4,
    "prefix_taus": [30.0, 120.0, 240.0, 480.0],
    "primary_clock": "restricted mean of normalized remaining energy in tau units",
    "primary_reference": "own long-window endpoint; fixed pair endpoint is a sensitivity control",
    "uncertainty_unit": "conditional processes and hierarchical initial-state/process bootstrap",
    "interpretation": "finite-window diagnostics; no infinite-mean or N-independent theorem",
}
KERNELS = (
    ("random", 1.0),
    ("opinion_random", 1.0),
    ("opinion_random", 4.0),
    ("structure_random_l0", 1.0),
    ("structure_random_l0", 4.0),
)


def design(profile="paper", horizon=None):
    """Return a frozen combined plan; smoke keeps every pipeline path short."""
    if profile not in ("smoke", "paper"):
        raise ValueError("evidence profile must be smoke or paper")
    small = profile == "smoke"
    if horizon is not None and (horizon <= 0 or (small and horizon > 1) or (not small and horizon < 120)):
        raise ValueError("smoke horizon must be <=1; formal horizon must be >=120")
    base = Parameters(
        alpha=0.02 if small else 0.01,
        bins=21 if small else 161,
        quadrature=3 if small else 9,
        horizon=(0.4 if small else 480.0) if horizon is None else horizon,
        record_tau=0.02 if small else 0.1,
    )
    tasks = {}

    def add(kind, p, role, **extra):
        p.validate()
        options = {
            "snapshot_taus": sorted(
                {
                    min(t, base.horizon)
                    for t in ([0.1, 0.2, base.horizon] if small else [4.0, 24.0, 30.0, 120.0, 240.0, base.horizon])
                }
            )
        }
        task = {"kind": kind, "parameters": asdict(p), "request_options": options, **extra}
        identifier = kind + "_" + fingerprint(task)[:20]
        if identifier not in tasks:
            tasks[identifier] = {**task, "id": identifier, "roles": []}
        tasks[identifier]["roles"].append(role)

    # Rates and confidence: the parameter-effects result. Random has no zeta effect.
    for rule, (rec, z), epsilon, ratio in product(
        ("hk", "deffuant"),
        KERNELS,
        (0.4,) if small else (0.2, 0.4, 0.8),
        (0.25, 2.0) if small else (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0),
    ):
        add(
            "measure",
            replace(base, dynamics=rule, recommender=rec, steepness=z, epsilon=epsilon, ratio=ratio),
            "effects",
        )
    # Fixed r, same numerical resolution and common scaled window: Eq54 evidence.
    for rule, (rec, z), ratio, alpha in product(
        ("hk", "deffuant"),
        KERNELS,
        (2.0,) if small else (0.25, 2.0, 20.0),
        (0.01, 0.02) if small else (0.0025, 0.005, 0.01, 0.02),
    ):
        add("measure", replace(base, dynamics=rule, recommender=rec, steepness=z, ratio=ratio, alpha=alpha), "scaling")
    # Separately vary space, quadrature and physical dt on declared slow-case anchors.
    for rule, (rec, z), (bins, quad, dt) in product(
        ("hk", "deffuant"),
        (KERNELS[2], KERNELS[4]),
        ((21, 3, 1.0), (31, 3, 1.0), (21, 5, 1.0), (21, 3, 0.5))
        if small
        else ((161, 9, 1.0), (81, 9, 1.0), (241, 9, 1.0), (161, 33, 1.0), (161, 9, 0.5)),
    ):
        add(
            "measure",
            replace(base, dynamics=rule, recommender=rec, steepness=z, ratio=2.0, bins=bins, quadrature=quad, dt=dt),
            "numerics",
        )
    # Nominal-N closure: identical initial law/resolution and fixed k,R.
    for rule, (rec, z), n in product(("hk", "deffuant"), KERNELS, (40, 80) if small else (500, 1000, 2000)):
        add("measure", replace(base, dynamics=rule, recommender=rec, steepness=z, ratio=2.0, population=n), "closure")
    # Initial states are independent; process branches share graph/opinions.
    for rule, (rec, z), ratio, n, initial, branch in product(
        ("hk", "deffuant"),
        KERNELS,
        (2.0,) if small else (0.25, 2.0, 20.0),
        (40, 80) if small else (500, 1000, 2000),
        range(2 if small else 4),
        range(2 if small else 4),
    ):
        p = replace(
            base,
            dynamics=rule,
            recommender=rec,
            steepness=z,
            ratio=ratio,
            population=n,
            alpha=0.02,
            bins=21 if small else 81,
        )
        add(
            "micro",
            p,
            "population",
            initial=initial,
            branch=branch,
            initial_seed=17000 + initial,
            process_seed=24000 + initial * 100 + branch,
            selection="weighted",
        )
    # Micro mesh/quad controls use exactly the same initial/process seeds.
    for rule, (bins, quad) in product(("hk", "deffuant"), ((31, 3), (21, 5)) if small else ((161, 9), (81, 33))):
        for initial, branch in product(range(2 if small else 4), range(2 if small else 4)):
            p = replace(
                base,
                dynamics=rule,
                recommender="opinion_random",
                steepness=4.0,
                ratio=2.0,
                population=40 if small else 500,
                alpha=0.02,
                bins=bins,
                quadrature=quad,
            )
            add(
                "micro",
                p,
                "micro_numerics",
                initial=initial,
                branch=branch,
                initial_seed=17000 + initial,
                process_seed=24000 + initial * 100 + branch,
                selection="weighted",
            )
    protocol = {**PROTOCOL}
    if small:
        protocol["prefix_taus"] = [t for t in (0.1, 0.2, base.horizon) if t <= base.horizon]
    else:
        protocol["prefix_taus"] = sorted({t for t in (30.0, 120.0, 240.0, base.horizon) if t <= base.horizon})
    return {"schema": 1, "profile": profile, "analysis_protocol": protocol, "tasks": list(tasks.values())}
