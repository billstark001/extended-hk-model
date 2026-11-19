"""Scientific comparison invariants for the fixed-ensemble mesh audit."""

from copy import deepcopy

import pytest

from experiments.paper_theory.evidence.grid_audit import comparison, design


def test_grid_design_keeps_one_physical_start_and_process_cohort_per_epsilon():
    plan = design()
    assert len(plan["tasks"]) == 192 + 36
    for epsilon in plan["epsilons"]:
        for initial in range(4):
            group = [t for t in plan["tasks"] if t["parameters"]["epsilon"] == epsilon and t["initial"] == initial]
            assert {t["initial_seed"] for t in group} == {17000 + initial}
            assert {t["parameters"]["ratio"] for t in group} == {10}
            assert len([t for t in group if t["kind"] == "micro_only"]) == 16
            assert {t["parameters"]["bins"] for t in group if t["kind"] == "pair_only"} == {81, 161, 241}


def example_records():
    p = dict(bins=3, epsilon=0.4, ratio=10, alpha=0.02, population=500, quadrature=33, dt=1)
    task = dict(parameters=p, initial=0, initial_seed=17000, process_seed=24000)
    pair = {"task": task, "result": {"initial_hash": "physical", "pair": {"frames": [
        {"step": 0, "rho": [1, 0, 0], "node_energy": 1},
        {"step": 1, "rho": [1, 0, 0], "node_energy": 0.5},
    ]}}}
    micros = []
    for index, mass in enumerate(([1, 0], [0, 1])):
        micros.append({"task": {**task, "parameters": {**p, "bins": 2}, "process_seed": 24000 + index},
                       "result": {"initial_hash": "physical", "frames": [
                           {"step": 0, "rho": mass, "exact_node_energy": 1},
                           {"step": 1, "rho": mass, "exact_node_energy": 0.5},
                       ]}})
    return micros, pair


def test_comparison_uses_conditional_mean_after_conservative_projection():
    micros, pair = example_records()
    row, fields = comparison(micros, pair, comparison_bins=2, bootstrap_samples=100)
    assert row["mean_rho_w1"] == pytest.approx(0.5)
    assert row["node_gap_of_mean_curve"] == 0
    assert sum(f["micro_mean_mass"] for f in fields) == pytest.approx(1)
    assert sum(f["map_mass"] for f in fields) == pytest.approx(1)


def test_comparison_rejects_changed_micro_start_or_observation_window():
    micros, pair = example_records()
    wrong = deepcopy(micros)
    wrong[0]["result"]["initial_hash"] = "different graph"
    with pytest.raises(ValueError, match="initial state"):
        comparison(wrong, pair, comparison_bins=2)
    wrong = deepcopy(micros)
    wrong[0]["result"]["frames"][-1]["step"] = 2
    with pytest.raises(ValueError, match="endpoint"):
        comparison(wrong, pair, comparison_bins=2)
