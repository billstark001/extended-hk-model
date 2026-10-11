"""Fresh design coverage and matching physical laws, with no old result fixture."""

import numpy as np

from experiments.paper_theory.evidence.design import design
from experiments.paper_theory.scaling.analyze import scaling_rows
from theory.parameters import Parameters
from theory.runtime.measure import measure_request


def test_formal_design_predeclares_long_tail_scaling_and_population_controls():
    plan = design("paper")
    tasks = plan["tasks"]
    assert len({t["id"] for t in tasks}) == len(tasks)
    assert {(t["parameters"]["recommender"], t["parameters"]["steepness"]) for t in tasks} == {
        ("random", 1.0),
        ("opinion_random", 1.0),
        ("opinion_random", 4.0),
        ("structure_random_l0", 1.0),
        ("structure_random_l0", 4.0),
    }
    scaling = [t for t in tasks if "scaling" in t["roles"]]
    assert {t["parameters"]["alpha"] for t in scaling} == {0.0025, 0.005, 0.01, 0.02}
    assert {t["parameters"]["horizon"] for t in tasks} == {480.0}
    population = [t for t in tasks if "population" in t["roles"]]
    assert {t["parameters"]["population"] for t in population} == {500, 1000, 2000}
    assert {(t["initial"], t["branch"]) for t in population} == {(i, b) for i in range(4) for b in range(4)}
    assert all(t["parameters"]["degree"] == 15 and t["parameters"]["recommendations"] == 10 for t in tasks)


def test_smoke_exercises_all_study_paths_without_formal_horizon():
    plan = design("smoke")
    assert {r for t in plan["tasks"] for r in t["roles"]} == {
        "effects",
        "scaling",
        "numerics",
        "closure",
        "population",
        "micro_numerics",
    }
    assert all(t["parameters"]["horizon"] <= 0.4 for t in plan["tasks"])
    assert plan["analysis_protocol"]["minimum_plateau_tau"] == 120


def test_snapshots_and_physical_step_are_explicit():
    p = Parameters(alpha=0.01, ratio=2, horizon=4, dt=0.5)
    request = measure_request("test", p, snapshot_taus=[1, 2, 4])
    assert request["snapshots"]["record_steps"] == [0, 1, 200, 400, 800]
    assert request["snapshots"]["edge"]
    assert request["dynamics"]["rewiring_rate"] == 0.02


def test_scaling_never_mixes_zeta_or_quadrature():
    case = {
        "dynamics": "hk",
        "recommender": "opinion_random",
        "epsilon": 0.4,
        "r": 2,
        "alpha": 0.02,
        "steepness": 1,
        "quadrature": 9,
        "tau": np.array([0, 1]),
        "node_energy": np.array([1, 0.5]),
        "edge_energy": np.array([1, 0.3]),
    }
    assert scaling_rows([case, {**case, "alpha": 0.01, "steepness": 4}]) == []
    assert scaling_rows([case, {**case, "alpha": 0.01, "quadrature": 33}]) == []
    assert len(scaling_rows([case, {**case, "alpha": 0.01}])) == 2
