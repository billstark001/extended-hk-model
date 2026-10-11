"""Operator parity, small-update and fixed-coefficient numerical task grids."""

from itertools import product

from experiments.paper_theory.runtime.tasks import DYNAMICS, RECS, request, task


def validation_cases():
    """Return measure/Fokker-Planck parity, physical small-update and fixed-coefficient numerical controls."""
    jobs = []
    # Equal fixed horizon and initial conditions within every measure/F-P pair.
    for eps, alpha, ratio, d, (rec, z), method in product(
        (0.4, 0.2, 0.8), (0.01, 0.05, 0.2), (0.25, 2, 4), DYNAMICS, RECS, ("measure", "fokker_planck")
    ):
        jobs.append(
            task(
                "operator",
                request(
                    d, rec, z, alpha=alpha, ratio=ratio, epsilon=eps, method=method, horizon=20, b=161, pathway=True
                ),
                tier="core" if eps == 0.4 else "confidence",
                meta={"scan": "validity"},
            )
        )
    # Fixed physical Delta=1; only numerical h and mesh change.
    for d, (rec, z), alpha, b, h in product(
        DYNAMICS, (RECS[0], RECS[2], RECS[4]), (0.02, 0.2), (81, 161, 241), (1, 0.5, 0.25)
    ):
        jobs.append(
            task(
                "operator",
                request(d, rec, z, alpha=alpha, ratio=2, b=b, h=h, method="fokker_planck", horizon=8),
                tier="numerical",
                physical_dt=1,
                meta={"scan": "fixed_coefficients"},
            )
        )
    # The physical Delta sweep is a different model and is labelled as such.
    for d, alpha, delta in product(DYNAMICS, (0.05, 0.2), (1, 0.5, 0.25, 0.125)):
        for method in ("measure", "fokker_planck"):
            jobs.append(
                task(
                    "operator",
                    request(d, alpha=alpha, ratio=2, h=delta, method=method, horizon=8, b=161),
                    tier="physical",
                    physical_dt=delta,
                    meta={"scan": "physical_limit"},
                )
            )
    # A numerical-error audit precedes interpreting the much larger validity scan.
    return [j for j in jobs if j["tier"] == "numerical"] + [j for j in jobs if j["tier"] != "numerical"]
