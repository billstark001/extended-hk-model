"""Select study-owned designs; batch labels are provenance, never module names."""

from itertools import product

from experiments.paper_theory.runtime.tasks import DYNAMICS, RECS, request, task

from .kernel_validation import design as kernel
from .micro_validation import design as micro
from .operator_validation import design as operator
from .release_clocks import design as clocks

FAMILIES = ("micro", "clocks", "operator", "kernel")
PROFILES = ("minimum", "core", "extended")
DEFAULT_PROFILE = "core"
TIERS = ("core", "heldout", "robustness", "grid", "confidence", "alpha", "numerical", "physical", "bridge")

_DESIGNS = {
    "micro": micro.validation_cases,
    "clocks": clocks.validation_cases,
    "operator": operator.validation_cases,
    "kernel": kernel.validation_cases,
}


def _smoke():
    jobs = []
    for d, (rec, z) in product(DYNAMICS, (RECS[0], RECS[2], RECS[4])):
        r = request(d, rec, z, alpha=0.1, ratio=2, b=21, horizon=0.4, n=80, degree=5, recommendations=3)
        jobs.append(task("micro", r, kind="micro", branches=4, meta={"smoke": True}))
    for method in ("measure", "fokker_planck"):
        parity = request("hk", alpha=0.1, ratio=2, b=21, method=method, horizon=0.4, pathway=True)
        parity["snapshots"]["final_rho"] = True
        parity["snapshots"]["final_edge"] = True
        jobs.append(task("operator", parity, kind="parity", meta={"smoke": True, "scan": "parity"}))
        for h in (1,) if method == "measure" else (1, 0.5):
            jobs.append(
                task(
                    "operator",
                    request("hk", alpha=0.1, ratio=2, b=21, h=h, method=method, horizon=0.4, pathway=True),
                    physical_dt=1,
                    meta={"smoke": True, "scan": "fixed_coefficients"},
                )
            )
    jobs.append(
        task(
            "clocks",
            request("hk", "opinion_random", 4, alpha=0.1, ratio=2, b=21, horizon=0.4, adaptive=True, pathway=True),
            meta={"smoke": True, "scan": "main"},
        )
    )
    for quad in (9, 33):
        jobs.append(
            task(
                "kernel",
                request(
                    alpha=0.2,
                    ratio=0,
                    b=21,
                    horizon=0.2,
                    degree=2,
                    recommendations=2,
                    law="short_multimodal",
                    confidence="center",
                    quadrature=quad,
                ),
                kind="frozen",
                law="short_multimodal",
                snapshots=False,
                meta={
                    "smoke": True,
                    "quadrature": quad,
                    "degree": 2,
                    "recommendations": 2,
                    "samples": 2000,
                    "scan": "hk_law",
                },
            )
        )
    for d, delta in product(DYNAMICS, (1, 0.5)):
        jobs.append(
            task(
                "kernel",
                request(d, alpha=0.2, ratio=0, b=21, h=delta, horizon=0.2, law="asymmetric", confidence="center"),
                kind="frozen",
                law="asymmetric",
                physical_dt=delta,
                snapshots=False,
                meta={"smoke": True, "scan": "mixed_term", "samples": 0},
            )
        )
    # A second B allows the conservative cross-grid analysis path to be checked.
    jobs.append(
        task(
            "operator",
            request("hk", alpha=0.1, ratio=2, b=31, method="fokker_planck", horizon=0.4),
            meta={"smoke": True, "scan": "fixed_coefficients"},
        )
    )
    for method in ("measure", "fokker_planck"):
        jobs.append(
            task(
                "operator",
                request("deffuant", alpha=0.1, ratio=2, b=21, method=method, horizon=0.4, pathway=True),
                meta={"smoke": True, "scan": "validity"},
            )
        )
    for rec in ("opinion_random", "structure_random_l0"):
        jobs.append(
            task(
                "micro",
                request("hk", rec, 4, alpha=0.1, ratio=2, b=21, horizon=0.4, n=80, degree=5, recommendations=3),
                kind="micro",
                selection="ranked",
                tier="bridge",
                meta={"smoke": True},
            )
        )
    jobs.append(
        task(
            "micro",
            request("hk", alpha=0.1, ratio=2, b=21, horizon=0.4, n=80, degree=5, recommendations=3),
            kind="micro",
            network="er",
            replicate=1,
            branches=4,
            tier="robustness",
            meta={"smoke": True},
        )
    )
    for law in ("bimodal", "asymmetric"):
        jobs.append(
            task(
                "micro",
                request("hk", alpha=0.1, ratio=2, b=21, horizon=0.4, n=80, degree=5, recommendations=3, law=law),
                kind="micro",
                law=law,
                meta={"smoke": True},
            )
        )
    return jobs


def _profile_jobs(family, profile):
    jobs = _DESIGNS[family]()
    if profile == "extended":
        return jobs
    if family == "micro":
        allowed = {"core", "heldout", "grid", "bridge"} if profile == "core" else {"core", "heldout"}
        return [j for j in jobs if j["tier"] in allowed]
    if family == "clocks":
        return [j for j in jobs if j["tier"] in ({"core", "confidence", "grid"} if profile == "core" else {"core"})]
    if family == "operator":
        jobs = [j for j in jobs if j["tier"] != "confidence"]
        if profile == "minimum":
            jobs = [
                j
                for j in jobs
                if j["tier"] != "numerical" or j["meta"]["b"] in (81, 161) and j["meta"]["h"] in (1, 0.5)
            ]
        return jobs
    return jobs


def generate(families=None, *, preset="validation", profile=DEFAULT_PROFILE, tiers=None, smoke=False):
    """Return ordered, identified tasks for a validation or ensembles preset.

    families/tiers filter the declared design and profile selects its full grid.
    smoke selects short interface checks rather than scientific production runs.
    Generated parameters, seeds and order determine frozen run identities.
    """
    if profile not in PROFILES or preset not in ("validation", "ensembles"):
        raise ValueError("unknown preset/profile")
    allowed = FAMILIES if preset == "validation" else ("micro", "clocks")
    selected = allowed if families is None else tuple(families)
    if len(set(selected)) != len(selected) or any(f not in allowed for f in selected):
        raise ValueError("families must be unique names available in this preset")
    if tiers is not None and any(t not in TIERS for t in tiers):
        raise ValueError("unknown tier")
    if preset == "validation":
        jobs = _smoke() if smoke else [j for f in selected for j in _profile_jobs(f, profile)]
    else:
        jobs = [
            j
            for f in selected
            for j in (
                getattr({"micro": micro, "clocks": clocks}[f], "ensemble_smoke")()
                if smoke
                else getattr({"micro": micro, "clocks": clocks}[f], "ensemble_cases")(profile)
            )
        ]
    if preset == "ensembles" and profile == "extended" and not smoke:
        jobs = [j for j in jobs if j["meta"].get("role") != "basin_long"] + [
            j for j in jobs if j["meta"].get("role") == "basin_long"
        ]
    jobs = [j for j in jobs if j["family"] in selected and (tiers is None or j["tier"] in tiers)]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job identities")
    return jobs
