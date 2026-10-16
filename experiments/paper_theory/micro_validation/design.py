"""Completed micro/pair and conditional-process audits retained by the supplement."""

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
            "smoke": False,
        },
    )
    # process=0 retains the original process RNG start exactly. Initial seed is
    # unaffected by process, mesh, quadrature, recommender or population size.
    job["payload"]["process_seed"] += process * PROCESS_SEED_STRIDE
    job["meta"].update(init_seed=job["payload"]["init_seed"], process_seed=job["payload"]["process_seed"])
    return identify(job)


def ensemble_cases():
    """Return the completed 236 micro/pair follow-up jobs cited by the supplement."""
    jobs = []
    grids = GRIDS
    initials = INITIAL_REPLICATES
    basin_processes = BASIN_PROCESS_REPLICATES
    clock_processes = CLOCK_PROCESS_REPLICATES
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
    for n, initial, process in product(POPULATIONS, initials, POPULATION_PROCESS_REPLICATES):
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
    return jobs
