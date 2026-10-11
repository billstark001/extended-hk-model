"""Matched small-update designs at fixed rewiring/influence ratio and scaled horizon."""

from itertools import product

from experiments.paper_theory.sweeps import task
from theory.parameters import Parameters


def designs(profile="paper"):
    """Yield scaling tasks across alpha with fixed ratio, confidence and numerical resolution for paper or smoke profiles."""
    _suite = "scaling"
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
    points = product(
        rules,
        recommendations,
        (0.4,),
        (0.0, 2.0) if small else (0.25, 2.0, 20.0),
        (0.01, 0.02) if small else (0.0025, 0.005, 0.01, 0.02),
    )
    for rule, rec, epsilon, ratio, alpha in points:
        yield task(
            _suite, Parameters(**common, dynamics=rule, recommender=rec, epsilon=epsilon, ratio=ratio, alpha=alpha)
        )
