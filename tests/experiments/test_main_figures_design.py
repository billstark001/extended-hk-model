"""Coverage and isolation contracts for the remaining manuscript main figures."""

import json
from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pytest

from common.experiments.plans import freeze_plan
from common.runtime.go import source_manifest
from experiments.paper_theory.evidence.analyze import clock_rows, key, pairing_excess
from experiments.paper_theory.evidence.bundle import Bundle, export_raw
from experiments.paper_theory.evidence.design import design
from experiments.paper_theory.evidence.figures import read_table
from experiments.paper_theory.evidence.fresh_reads import FreshReadPolicy
from theory.parameters import Parameters
from theory.release import release_clock


def test_confidence_and_observation_windows_are_distinct_ensemble_groups():
    baseline = Parameters(epsilon=0.4, horizon=120)
    assert key(asdict(baseline)) != key(asdict(replace(baseline, horizon=480)))
    assert key(asdict(baseline)) != key(asdict(replace(baseline, epsilon=0.2)))


def test_figure_tables_never_pool_different_confidence_radii(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "population.csv").write_text("epsilon,population,node_mean\n0.2,500,1\n0.4,500,2\n0.8,500,3\n")
    rows = read_table(tables / "population.csv", epsilon=0.4)
    assert len(rows) == 1 and rows[0]["node_mean"] == 2


def test_formal_mesoscopic_stopping_preserves_scaled_time_across_controls():
    plan = design("main")
    for task in plan["tasks"]:
        stopping = task["request_options"].get("stopping")
        if task["kind"] == "micro":
            assert stopping is None  # Matched ensembles still share a fixed observation grid.
            continue
        p = Parameters(**task["parameters"])
        tau_step = p.alpha * p.dt
        assert stopping["mode"] == "state_and_energy"
        assert np.isclose(stopping["minimum_steps"] * tau_step, 120)
        assert np.isclose(stopping["patience_steps"] * tau_step, 40)
        assert np.isclose(stopping["state_l1_tolerance"] / tau_step, 1e-5)
        assert np.isclose(stopping["energy_absolute_tolerance"] / tau_step, 1e-7)
    assert all("stopping" not in t["request_options"] for t in design("main_smoke")["tasks"])


def test_early_stopped_area_retains_observed_endpoint_without_fabricating_tail():
    protocol = design("main")["analysis_protocol"]
    time = np.linspace(0, 160.2, 802)
    energy = np.exp(-time / 4)
    rows = clock_rows(time, {"node": energy}, protocol, {})
    endpoints = [r for r in rows if r["is_observed_endpoint"]]
    assert len(endpoints) == 1
    assert endpoints[0]["horizon_tau"] == time[-1]
    assert np.isclose(endpoints[0]["mean_tau"], release_clock(time, energy).area)
    assert endpoints[0]["plateau_confirmed"]
    assert set(float(t) for t in endpoints[0]["unobserved_prefix_taus"].split(",")) == {240, 480}
    assert all(r["horizon_tau"] <= time[-1] for r in rows)


def test_pruned_main_plans_have_fixed_fingerprints():
    from common.io.json import fingerprint

    assert fingerprint(design("main")) == "3843c2cade76e352caa58481a08c2f2f5f38f67eee3969096d06785f02af7015"
    assert fingerprint(design("main_smoke")) == "70dca64c55bf181b708e61590479e279477a3320d0b4aa472fa279792d8603a2"
    for profile in ("paper", "smoke"):
        with pytest.raises(ValueError, match="main"):
            design(profile)


def test_shared_pair_dependencies_preserve_all_main_process_branches():
    from common.io.json import fingerprint
    from theory.runtime.population import pair_identity

    tasks = [t for t in design("main")["tasks"] if t["kind"] == "micro"]
    cohorts = Counter(fingerprint(pair_identity(t, {"source": "frozen"})) for t in tasks)
    assert len(cohorts) == 228
    assert sum(cohorts.values()) == 1056
    assert set(cohorts.values()) == {4, 8}


def test_main_coverage_matches_captions_and_two_independent_initial_graphs():
    plan = design("main")
    tasks = plan["tasks"]
    assert len({t["id"] for t in tasks}) == len(tasks)
    assert {t["parameters"]["epsilon"] for t in tasks} == {0.2, 0.4, 0.8}
    assert {t["parameters"]["horizon"] for t in tasks} == {120.0, 480.0}
    assert all(t["parameters"]["horizon"] == 480 for t in tasks if "population" not in t["roles"])
    effects = [t for t in tasks if "effects" in t["roles"]]
    assert len(effects) == 3 * 2 * 5 * 11
    assert {t["parameters"]["ratio"] for t in effects} >= {0, 0.25, 2, 10, 32}
    matched = [t for t in tasks if "matched" in t["roles"]]
    assert len(matched) == 3 * 2 * 3 * 2 * 8
    assert {t["parameters"]["population"] for t in matched} == {500}
    assert {t["parameters"]["bins"] for t in matched} == {161}
    assert {t["parameters"]["quadrature"] for t in matched} == {33}
    assert all(
        v == 8
        for v in Counter(
            (t["parameters"]["dynamics"], t["parameters"]["recommender"], t["parameters"]["epsilon"], t["initial"])
            for t in matched
        ).values()
    )
    population = [t for t in tasks if "population" in t["roles"]]
    assert len(population) == 3 * 2 * 2 * 4 * 4 * 4
    assert {t["parameters"]["bins"] for t in population} == {161}
    assert {t["parameters"]["quadrature"] for t in population} == {33}
    assert {(t["parameters"]["recommender"], t["parameters"]["steepness"]) for t in population} == {
        ("random", 1.0),
        ("opinion_random", 4.0),
    }
    assert {t["parameters"]["horizon"] for t in population} == {120.0}
    assert {t["parameters"]["population"] for t in population} == {250, 500, 1000, 2000}
    scaling = [t for t in tasks if "scaling" in t["roles"]]
    assert {t["parameters"]["alpha"] for t in scaling} == {0.04, 0.02, 0.01, 0.005, 0.0025}
    assert all(t["parameters"]["ratio"] * t["parameters"]["alpha"] <= 1 for t in tasks)
    with pytest.raises(ValueError):
        design("main", horizon=30)


def test_main_smoke_retains_all_five_research_topics():
    plan = design("main_smoke")
    assert {r for t in plan["tasks"] for r in t["roles"]} == {"effects", "scaling", "numerics", "population", "matched"}
    assert all(t["parameters"]["horizon"] == 0.4 for t in plan["tasks"])
    assert json.loads(json.dumps(plan)) == plan


def test_read_policy_rejects_old_results_and_symlink_targets(tmp_path):
    work, publish, source, old = [tmp_path / p for p in ("work", "publish", "source", "old")]
    for root in (work, source, old):
        root.mkdir()
    old_data = old / "existing.csv"
    old_data.write_text("a,b\n1,2\n")
    (work / "pretend_new.csv").symlink_to(old_data)
    policy = FreshReadPolicy(work, publish, [source], [])
    assert policy.classify(work / "new.npz") == "generated"
    assert policy.classify(source / "src" / "operator.py") == "source"
    assert policy.classify(source / "outputs" / "previous.npz") is None
    assert policy.classify(old_data) is None
    assert policy.classify(work / "pretend_new.csv") is None
    with pytest.raises(PermissionError):
        policy.audit("open", (str(work / "pretend_new.csv"), "r", 0))


def test_pairing_reference_uses_actual_marginals_and_flattened_fields():
    independent = np.outer([0.2, 0.8], [0.7, 0.3])
    frame = {"rho": [0.2, 0.8], "edge": independent.ravel()}
    assert abs(pairing_excess(frame, 0.1)) < 1e-14
    diagonal = {"rho": [0.5, 0.5], "edge": [0.5, 0, 0, 0.5]}
    assert pairing_excess(diagonal, 0.1) == 0.5


def test_mean_integral_does_not_clip_negative_remaining_energy():
    clock = release_clock([0, 1, 2, 3], [1, -0.2, -0.2, 0], plateau=0)
    assert np.isclose(clock.area, 0.1)
    assert clock.negative_area > 0


def test_generated_appledouble_metadata_is_not_a_result_archive(tmp_path):
    task = design("smoke_scaling")["tasks"][0]
    freeze_plan(tmp_path / "plan.json", {"tasks": [task]})
    raw = tmp_path / "measure" / "raw"
    raw.mkdir(parents=True)
    np.savez(
        raw / (task["id"] + ".npz"),
        metadata=json.dumps(task),
        diagnostics=json.dumps({"converged": False}),
        tau=np.array([0.0, 1.0]),
    )
    (raw / ("._" + task["id"] + ".npz")).write_bytes(b"AppleDouble metadata, not NumPy")
    output = tmp_path / "local"
    export_raw(tmp_path, output)
    assert list(Bundle(output).manifest["records"]) == [task["id"]]


def test_source_hashing_skips_historical_output_directories(tmp_path):
    (tmp_path / "go.mod").write_text("module test\n")
    (tmp_path / "go.sum").write_text("")
    (tmp_path / "core.go").write_text("package test\n")
    old = tmp_path / "outputs"
    old.mkdir()
    (old / "not_production.go").write_text("old generated data")
    assert set(source_manifest(tmp_path)) == {"core.go", "go.mod", "go.sum"}
