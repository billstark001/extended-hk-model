"""Frozen design, runtime schema and explicit no-execution freeze boundary."""

import ast
from pathlib import Path

import pytest

from experiments.paper_lifted.__main__ import parser
from experiments.paper_lifted.evidence.design import formal_design, validation_design
from lifted.runtime.costs import task_cost
from lifted.runtime.protocol import model_request, request, stream_seed


def test_formal_settings_match_theory_and_dependencies_have_earlier_stages():
    plan = formal_design(initials=4, mothers=2, micro_paths=128, meso_paths=16, budget={})
    tasks = {t["id"]: t for t in plan["tasks"]}
    earlier = set()
    for stage in plan["stages"]:
        for key in stage["task_ids"]:
            task = tasks[key]
            assert task["source_task"] is None or task["source_task"] in earlier
            p = task["parameters"]
            assert (
                p["dynamics"],
                p["recommender"],
                p["alpha"],
                p["epsilon"],
                p["degree"],
                p["recommendations"],
                p["quadrature"],
            ) == ("hk", "random", 0.02, 0.4, 15, 10, 33)
        earlier.update(stage["task_ids"])
    assert len(tasks) == len(plan["tasks"])
    assert all(t["source_step"] != 0 or t["mother"] == 0 for t in tasks.values() if t["kind"] == "micro_probe")
    controls = [tasks[i] for s in plan["stages"] if s["name"] == "noise_controls" for i in s["task_ids"]]
    assert controls and all(
        t["horizon_steps"] == 6000 and len(t["replicates"]) <= 8 and t["cohort"] == "mechanism" for t in controls
    )


def test_validation_profiles_are_bounded_and_do_not_contain_formal_work():
    smoke = validation_design("smoke")
    assert max(t["horizon_steps"] for t in smoke["tasks"]) <= 24
    assert all(t["cohort"] == "smoke" for t in smoke["tasks"])
    pilot = validation_design("pilot")
    assert {t["parameters"]["bins"] for t in pilot["tasks"]} >= {81, 161, 241}
    assert {t["source_step"] for t in pilot["tasks"] if t["kind"] == "micro_probe"} >= {0, 200, 1500}
    with pytest.raises(ValueError, match="pilot"):
        validation_design("formal")


def test_branch_randomness_depends_on_identity_not_workers_or_chunking(tmp_path):
    plan = validation_design("smoke")
    task = next(t for t in plan["tasks"] if t["kind"] == "micro_probe")
    a = request(task, {"checkpoint": "x", "checkpoint_sha256": "a"}, tmp_path, source_kind="micro")
    b = request(
        {**task, "replicates": [task["replicates"][2]]},
        {"checkpoint": "x", "checkpoint_sha256": "a"},
        tmp_path,
        source_kind="micro",
    )
    assert a["seeds"][2] == b["seeds"][0]
    assert len({r["value"] for r in a["seeds"]}) == len(task["replicates"])
    assert stream_seed(1, "a") != stream_seed(1, "b")
    assert model_request(task)["major_cluster_mass"] < 1 / task["parameters"]["population"]


def test_cost_model_extends_censored_horizon_instead_of_treating_short_probe_as_absorbed():
    task = next(t for t in validation_design("smoke")["tasks"] if t["kind"] == "micro_probe")
    p = task["parameters"]
    row = dict(
        kind="micro_probe",
        variant="stochastic",
        ratio=p["ratio"],
        bins=p["bins"],
        population=p["population"],
        paths=4,
        executed_steps=40,
        horizon_steps=10,
        elapsed_seconds=4,
        wall_seconds=4.1,
        unresolved=4,
    )
    short, _ = task_cost({**task, "horizon_steps": 10}, [row])
    long, _ = task_cost({**task, "horizon_steps": 100}, [row])
    assert long > 9 * short


def test_only_cli_main_and_source_packages_do_not_import_experiments():
    root = Path(__file__).resolve().parents[2]
    for source in (root / "src/lifted").rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("experiments")
    found = []
    for source in (root / "experiments/paper_lifted").rglob("*.py"):
        if any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in ast.walk(ast.parse(source.read_text()))):
            found.append(source.name)
    assert found == ["__main__.py"]
    parsed = parser().parse_args(["freeze", "--work-dir", "new", "--pilot-dir", "pilot"])
    assert parsed.action == "freeze"
