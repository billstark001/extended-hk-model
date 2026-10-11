"""Matched microscopic/pair population designs with nested initial/process replicates."""

from dataclasses import asdict
from itertools import product

import numpy as np

from theory.parameters import Parameters


def designs(profile="paper"):
    """Yield uniform-law k=15/R=10 tasks; independent initial seeds and conditional process branches are separate statistical units."""
    small = profile == "smoke"
    if profile not in ("smoke", "paper"):
        raise ValueError("unknown profile")
    recs = ("random",) if small else ("random", "opinion_random", "structure_random_l0")
    ns = (40, 80) if small else (500, 1000, 2000)
    for rule, rec, ratio, n, initial, branch in product(
        ("hk", "deffuant"),
        recs,
        (2.0,) if small else (0.25, 2.0, 20.0),
        ns,
        range(1 if small else 4),
        range(2 if small else 4),
    ):
        p = Parameters(
            dynamics=rule,
            recommender=rec,
            ratio=ratio,
            population=n,
            alpha=0.02,
            bins=21 if small else 81,
            quadrature=3 if small else 9,
            horizon=0.2 if small else 30.0,
            record_tau=0.02 if small else 0.25,
        )
        p.validate()
        yield {
            "id": f"micro_{rule}_{rec}_r{ratio:g}_n{n}_i{initial}_p{branch}",
            "parameters": asdict(p),
            "initial_seed": 17000 + initial,
            "process_seed": int(
                np.random.SeedSequence([24000 + initial, branch]).generate_state(1, dtype=np.uint64)[0]
            ),
            "initial": initial,
            "branch": branch,
            "selection": "weighted",
        }
