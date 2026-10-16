"""Completed paired-clock, crossing and long-tail audits retained by the supplement."""

from copy import deepcopy
from itertools import product

from experiments.paper_theory.protocols import (
    ANALYSIS,
    CLOCK_HORIZON,
    CROSSING_RATIOS,
    EXTENSION_CASES,
    EXTENSION_HORIZON,
    QUADRATURES,
)
from experiments.paper_theory.runtime import tasks as base
from experiments.paper_theory.runtime.tasks import DYNAMICS, RECS, identify, request, task


def validation_cases():
    """Return rate strips and confidence/alpha/grid controls for direct paired energy clocks."""
    jobs = []
    # At least one complete rate strip before confidence/alpha robustness.
    for ratio, d, (rec, z) in product((0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32), DYNAMICS, RECS):
        jobs.append(
            task(
                "clocks",
                request(d, rec, z, ratio=ratio, horizon=120, b=161, adaptive=True, pathway=True),
                meta={"scan": "main"},
            )
        )
    for eps, ratio, d, (rec, z) in product((0.2, 0.4, 0.8), (0.25, 2, 16), DYNAMICS, RECS):
        jobs.append(
            task(
                "clocks",
                request(d, rec, z, epsilon=eps, ratio=ratio, horizon=120, b=161, adaptive=True, pathway=True),
                tier="confidence",
                meta={"scan": "confidence"},
            )
        )
    for d, (rec, z), b in product(DYNAMICS, (RECS[0], RECS[2], RECS[4]), (81, 241)):
        jobs.append(
            task(
                "clocks",
                request(d, rec, z, ratio=2, horizon=120, b=b, adaptive=True, pathway=True),
                tier="grid",
                meta={"scan": "grid"},
            )
        )
    return jobs


def ensemble_case(
    d, rec, z, *, ratio, epsilon=0.45, b=161, quadrature=9, role="crossing", horizon=CLOCK_HORIZON, alpha=0.02
):
    """Construct one identified long-horizon clock task, preserving strict joint stopping and independent energy-check thresholds."""
    request = base.request(
        d, rec, z, alpha=alpha, ratio=ratio, epsilon=epsilon, b=b, quadrature=quadrature, horizon=horizon, adaptive=True
    )
    job = base.task(
        "clocks",
        request,
        tier="core" if role != "extension" else "confidence",
        meta={
            "study": "followup_20261007",
            "role": role,
            "scan": role,
            "quadrature": quadrature,
            "maximum_tau": horizon,
            "smoke": False,
        },
    )
    # Retain the strict joint stop online. Energy-only confirmations are checked
    # independently on all saved scalar checkpoints using these same thresholds.
    job["meta"]["energy_check"] = deepcopy(request["stopping"])
    job["payload"]["snapshot_steps"] = sorted(
        {
            0,
            request["steps"],
            *(
                min(request["steps"], round(t / (alpha * request["dt"])))
                for t in ANALYSIS["snapshot_taus"]
                if t <= horizon
            ),
        }
    )
    return identify(job)


def ensemble_cases():
    """Return the completed 110 extension, crossing and anchor jobs."""
    jobs = []
    for d, rec, z, ratio, eps in EXTENSION_CASES:
        jobs.append(ensemble_case(d, rec, z, ratio=ratio, epsilon=eps, role="extension", horizon=EXTENSION_HORIZON))
    for d, z, b, quad in product(base.DYNAMICS, (1, 4), (161, 241), QUADRATURES):
        for ratio in CROSSING_RATIOS[z]:
            jobs.append(ensemble_case(d, "opinion_random", z, ratio=ratio, b=b, quadrature=quad))
    for d, (rec, z), b, quad in product(base.DYNAMICS, (base.RECS[0], base.RECS[4]), (161, 241), QUADRATURES):
        jobs.append(ensemble_case(d, rec, z, ratio=2, b=b, quadrature=quad, role="anchor"))
    return jobs
