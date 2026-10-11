"""Micro/pair, conditional-process and nominal-population closure designs."""

from itertools import product

from experiments.paper_theory.protocols import (
    BASIN_PROCESS_REPLICATES,
    CLOCK_INITIAL_REPLICATES,
    CLOCK_PROCESS_REPLICATES,
    ENSEMBLE_BINS,
    ENSEMBLE_QUADRATURE,
    GRIDS,
    HOLDOUT_FAILURES,
    INITIAL_REPLICATES,
    MICRO_HORIZON,
    POPULATION_PROCESS_REPLICATES,
    POPULATIONS,
    PROCESS_SEED_STRIDE,
    QUADRATURES,
)
from experiments.paper_theory.runtime import tasks as base
from experiments.paper_theory.runtime.tasks import DYNAMICS, RECS, identify, request, task
from experiments.paper_theory.sweeps import task as sweep_task
from theory.parameters import Parameters


def validation_cases():
    """Return matched micro/pair validation tasks with held-out laws, kernels, populations and conditional audits."""
    jobs = []
    # Interleave replicate blocks: a budget-limited run covers every cell first.
    for rep, d, (rec, z), ratio in product(range(4), DYNAMICS, RECS, (0.25, 2, 10)):
        r = request(d, rec, z, ratio=ratio, horizon=30)
        jobs.append(task("micro", r, kind="micro", replicate=rep, branches=64 if rep == 0 else 0))
    for law, rep, d, (rec, z) in product(("bimodal", "asymmetric"), range(2), DYNAMICS, RECS):
        jobs.append(
            task(
                "micro",
                request(d, rec, z, ratio=2, law=law, horizon=30),
                kind="micro",
                law=law,
                replicate=rep + 20,
                tier="heldout",
                branches=32 if rep == 0 else 0,
            )
        )
    for network, n, rep, d, (rec, z), ratio in product(
        ("er", "fixed"), (500,), range(2), DYNAMICS, (RECS[0], RECS[2], RECS[4]), (0.25, 10)
    ):
        if network == "fixed":
            n = 1000
        jobs.append(
            task(
                "micro",
                request(d, rec, z, ratio=ratio, n=n, horizon=30),
                kind="micro",
                network=network,
                replicate=rep + 40,
                tier="robustness",
            )
        )
    for b, d, (rec, z) in product((41, 161), DYNAMICS, (RECS[0], RECS[2], RECS[4])):
        jobs.append(task("micro", request(d, rec, z, ratio=2, b=b, horizon=30), kind="micro", replicate=0, tier="grid"))
    for rep, d, (rec, z), ratio in product(range(2), DYNAMICS, (RECS[0], RECS[2], RECS[4]), (0.5, 6)):
        jobs.append(
            task(
                "micro",
                request(d, rec, z, alpha=0.035, ratio=ratio, horizon=30),
                kind="micro",
                replicate=rep + 60,
                tier="heldout",
                meta={"holdout": "parameter"},
            )
        )
    for d, (rec, z), ratio in product(DYNAMICS, (RECS[2], RECS[4]), (0.25, 2, 10)):
        jobs.append(
            task(
                "micro",
                request(d, rec, z, ratio=ratio, horizon=30),
                kind="micro",
                selection="ranked",
                replicate=0,
                tier="bridge",
            )
        )
    return jobs


def ensemble_case(
    d="hk",
    rec="random",
    z=1,
    *,
    ratio=10,
    initial=0,
    process=0,
    n=500,
    b=ENSEMBLE_BINS,
    quadrature=ENSEMBLE_QUADRATURE,
    alpha=0.02,
    role="basin_ensemble",
    tier="core",
    horizon=MICRO_HORIZON,
    smoke=False,
):
    """Construct one identified micro/pair process branch with explicit initial/process replicate metadata."""
    request = base.request(d, rec, z, alpha=alpha, ratio=ratio, n=n, b=b, quadrature=quadrature, horizon=horizon)
    job = base.task(
        "micro",
        request,
        kind="micro",
        replicate=initial,
        tier=tier,
        meta={
            "study": "followup_20261007",
            "role": role,
            "initial_replicate": initial,
            "process_replicate": process,
            "quadrature": quadrature,
            "smoke": smoke,
        },
    )
    # process=0 retains the original process RNG start exactly. Initial seed is
    # unaffected by process, mesh, quadrature, recommender or population size.
    job["payload"]["process_seed"] += process * PROCESS_SEED_STRIDE
    job["meta"].update(init_seed=job["payload"]["init_seed"], process_seed=job["payload"]["process_seed"])
    return identify(job)


def ensemble_cases(profile):
    """Return conditional process ensembles and resolution/population controls for the requested production profile."""
    jobs = []
    grids = (81, 161) if profile == "minimum" else GRIDS
    initials = (0, 2) if profile == "minimum" else INITIAL_REPLICATES
    basin_processes = tuple(range(8)) if profile == "minimum" else BASIN_PROCESS_REPLICATES
    clock_processes = tuple(range(4)) if profile == "minimum" else CLOCK_PROCESS_REPLICATES
    for initial, b, quad in product(initials, grids, QUADRATURES):
        jobs.append(ensemble_case(initial=initial, b=b, quadrature=quad, role="basin_grid", tier="grid"))
    for initial, process in product(initials, basin_processes):
        if process == 0:
            continue
        jobs.append(ensemble_case(initial=initial, process=process))
    for d, (rec, z), initial, process in product(
        base.DYNAMICS, (base.RECS[0], base.RECS[2], base.RECS[4]), CLOCK_INITIAL_REPLICATES, clock_processes
    ):
        jobs.append(ensemble_case(d, rec, z, ratio=2, initial=initial, process=process, role="clock_ensemble"))
    for n, initial, process in product(
        (1000,) if profile == "minimum" else POPULATIONS, initials, POPULATION_PROCESS_REPLICATES
    ):
        jobs.append(ensemble_case(n=n, initial=initial, process=process, role="population", tier="robustness"))
    for d, rec, z, ratio, initial in HOLDOUT_FAILURES:
        for b, quad in product(grids, QUADRATURES):
            jobs.append(
                ensemble_case(
                    d,
                    rec,
                    z,
                    ratio=ratio,
                    initial=initial,
                    alpha=0.035,
                    b=b,
                    quadrature=quad,
                    role="holdout_grid",
                    tier="heldout",
                )
            )
    if profile == "extended":
        for initial in INITIAL_REPLICATES:
            jobs.append(ensemble_case(initial=initial, horizon=60, role="basin_long", tier="robustness"))
    return jobs


def ensemble_smoke():
    """Return short conditional-ensemble tasks exercising branch, mesh and quadrature handling."""
    jobs = []
    for initial, process in product((0, 1), (0, 1, 2)):
        jobs.append(
            ensemble_case(
                initial=initial,
                process=process,
                n=80,
                b=21,
                quadrature=9,
                horizon=0.4,
                ratio=2,
                role="clock_ensemble",
                smoke=True,
            )
        )
    for b, quad in ((21, 33), (31, 9), (31, 33)):
        jobs.append(ensemble_case(n=80, b=b, quadrature=quad, horizon=0.4, role="basin_grid", smoke=True))
    jobs.append(ensemble_case(n=120, b=21, quadrature=9, horizon=0.4, role="population", smoke=True))
    return jobs


def closure_designs(profile="paper"):
    """Yield pair-closure tasks varying nominal N while retaining other physical parameters; this does not establish microscopic N convergence."""
    _suite = "population_closure"
    if profile not in ("smoke", "paper"):
        raise ValueError("unknown profile")
    small = profile == "smoke"
    common = {
        "bins": 21 if small else 161,
        "quadrature": 3 if small else 9,
        "horizon": 0.2 if small else 30.0,
        "record_tau": 0.01 if small else 0.05,
    }
    rules = ("hk", "deffuant")
    recommendations = ("random",) if small else ("random", "opinion_random", "structure_random_l0")
    for rule, rec, ratio, n in product(rules, recommendations, (2.0,), (500, 1000, 2000)):
        p = Parameters(**common, dynamics=rule, recommender=rec, ratio=ratio, population=n)
        yield sweep_task(_suite, p)
    return
