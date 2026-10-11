"""Design for the three remaining main figures, independent of archived data."""

from dataclasses import asdict, replace
from itertools import product
from math import ceil

from common.io.json import fingerprint
from theory.parameters import Parameters

from .paper_design import KERNELS, PROTOCOL


def stopping_options(p):
    """Check joint state/energy stationarity in scaled time, independent of dt/alpha."""
    tau_step = p.alpha * p.dt
    return {
        "mode": "state_and_energy",
        "minimum_steps": ceil(PROTOCOL["minimum_plateau_tau"] / tau_step),
        "check_every": max(1, ceil(0.2 / tau_step)),
        "patience_steps": ceil(2 * PROTOCOL["plateau_window_tau"] / tau_step),
        "state_l1_tolerance": 1e-5 * tau_step,
        "energy_absolute_tolerance": 1e-7 * tau_step,
        "energy_relative_tolerance": 1e-5 * tau_step,
    }


def design(smoke=False, horizon=None):
    """Freeze physical settings, matched seeds, tails and numerical controls."""
    horizon = (0.4 if smoke else 480.0) if horizon is None else horizon
    if horizon <= 0 or (smoke and horizon > 1) or (not smoke and horizon < 480):
        raise ValueError("main smoke horizon <=1; formal horizon >=480 required")
    bins, quad = (21, 3) if smoke else (161, 33)
    base = Parameters(
        alpha=0.02, epsilon=0.4, bins=bins, quadrature=quad, horizon=horizon, record_tau=0.02 if smoke else 0.1
    )
    snapshots = sorted(
        {min(t, horizon) for t in ([0.1, 0.2, horizon] if smoke else [4.0, 24.0, 30.0, 120.0, 240.0, 480.0, horizon])}
    )
    tasks = {}
    epsilons = (0.4,) if smoke else (0.2, 0.4, 0.8)

    def add(kind, p, role, **extra):
        for epsilon in epsilons:
            setting = replace(p, epsilon=epsilon)
            setting.validate()
            options = {"snapshot_taus": snapshots}
            if kind == "measure" and not smoke:
                options["stopping"] = stopping_options(setting)
            task = dict(kind=kind, parameters=asdict(setting), request_options=options, **extra)
            identifier = kind + "_" + fingerprint(task)[:20]
            if identifier not in tasks:
                tasks[identifier] = dict(task, id=identifier, roles=[])
            if role not in tasks[identifier]["roles"]:
                tasks[identifier]["roles"].append(role)

    rates = (0.0, 0.25, 2.0, 4.0) if smoke else (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 16.0, 32.0)
    for rule, (rec, z), ratio in product(("hk", "deffuant"), KERNELS, rates):
        add("measure", replace(base, dynamics=rule, recommender=rec, steepness=z, ratio=ratio), "effects")
    for rule, (rec, z), ratio, alpha in product(
        ("hk", "deffuant"),
        KERNELS,
        (2.0,) if smoke else (0.25, 2.0, 10.0),
        (0.01, 0.02) if smoke else (0.04, 0.02, 0.01, 0.005, 0.0025),
    ):
        add("measure", replace(base, dynamics=rule, recommender=rec, steepness=z, ratio=ratio, alpha=alpha), "scaling")
    controls = ((bins, quad, 1.0), (31 if smoke else 81, quad, 1.0), (bins, 5 if smoke else 9, 1.0), (bins, quad, 0.5))
    if not smoke:
        controls += ((241, quad, 1.0),)
    for rule, (rec, z), (b, q, dt) in product(("hk", "deffuant"), (KERNELS[2], KERNELS[4]), controls):
        add(
            "measure",
            replace(base, dynamics=rule, recommender=rec, steepness=z, bins=b, quadrature=q, dt=dt),
            "numerics",
        )
    populations = (40, 80) if smoke else (250, 500, 1000, 2000)
    matched_population = 40 if smoke else 500
    population_kernels = KERNELS if smoke else (KERNELS[0], KERNELS[2])
    population_horizon = horizon if smoke else min(120.0, horizon)
    for rule, (rec, z), n in product(("hk", "deffuant"), KERNELS, populations):
        add("measure", replace(base, dynamics=rule, recommender=rec, steepness=z, population=n), "closure")
    for rule, (rec, z), n, initial, branch in product(
        ("hk", "deffuant"), population_kernels, populations, range(2 if smoke else 4), range(2 if smoke else 4)
    ):
        p = replace(
            base,
            dynamics=rule,
            recommender=rec,
            steepness=z,
            population=n,
            bins=bins,
            quadrature=quad,
            horizon=population_horizon,
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
    # Exactly two initial graphs and eight conditional processes for main validation.
    for rule, (rec, z), initial, branch in product(
        ("hk", "deffuant"), (KERNELS[0], KERNELS[2], KERNELS[4]), range(2), range(2 if smoke else 8)
    ):
        p = replace(base, dynamics=rule, recommender=rec, steepness=z, population=matched_population)
        add(
            "micro",
            p,
            "matched",
            initial=initial,
            branch=branch,
            initial_seed=17000 + initial,
            process_seed=24000 + initial * 100 + branch,
            selection="weighted",
        )
    protocol = dict(
        PROTOCOL,
        prefix_taus=[t for t in snapshots if t >= (0.1 if smoke else 30)],
        numerical_baseline={"bins": bins, "quadrature": quad, "dt": 1.0},
        main_population_bins=bins,
        main_population_quadrature=quad,
        population_horizon_tau=population_horizon,
        population_kernels=[list(k) for k in population_kernels],
        population_ratio=2.0,
        population_claim="finite-window structure robustness; structural-score size audit is separate",
        main_snapshot_tau=0.2 if smoke else 24.0,
        baseline_alpha=0.02,
        baseline_epsilon=0.4,
        confidence_radii=list(epsilons),
        matched_population=matched_population,
        matched_bins=bins,
        matched_quadrature=quad,
        minimum_plateau_tau=120.0,
        adaptive_measure_stopping=not smoke,
        maximum_horizon_tau=horizon,
        measure_stopping="joint state/energy rates; tau>=120, stable for 40 tau; no extrapolated tail",
        primary_clock="unclipped finite-window residual area; full means require tail confirmation",
    )
    return dict(
        schema=1, profile="main_smoke" if smoke else "main", analysis_protocol=protocol, tasks=list(tasks.values())
    )
