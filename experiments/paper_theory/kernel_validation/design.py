"""Frozen-kernel and mixed-increment tasks with quadrature/Monte Carlo controls."""

from itertools import product

from experiments.paper_theory.runtime.tasks import DYNAMICS, RECS, request, task


def validation_cases():
    """Return frozen finite-list and mixed-increment audit tasks with declared quadrature and Monte Carlo controls."""
    jobs = []
    for law, counts, (rec, z), quad in product(
        ("uniform", "asymmetric", "short_multimodal"), ((2, 2), (15, 10)), (RECS[0], RECS[2]), (9, 33)
    ):
        r = request(
            "hk",
            rec,
            z,
            alpha=0.2,
            ratio=0,
            b=81,
            horizon=0.2,
            law=law,
            degree=counts[0],
            recommendations=counts[1],
            quadrature=quad,
            confidence="center",
        )
        jobs.append(
            task(
                "kernel",
                r,
                kind="frozen",
                law=law,
                budgets=False,
                snapshots=False,
                meta={
                    "quadrature": quad,
                    "degree": counts[0],
                    "recommendations": counts[1],
                    "samples": 200000,
                    "scan": "hk_law",
                },
            )
        )
    for d, delta in product(DYNAMICS, (1, 0.5, 0.25, 0.125)):
        r = request(d, alpha=0.2, ratio=0, b=81, h=delta, horizon=0.2, law="asymmetric", confidence="center")
        jobs.append(
            task(
                "kernel",
                r,
                kind="frozen",
                law="asymmetric",
                physical_dt=delta,
                budgets=False,
                snapshots=False,
                meta={"scan": "mixed_term", "samples": 0},
            )
        )
    return jobs
