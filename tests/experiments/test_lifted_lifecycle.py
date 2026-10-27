"""Freeze never executes, and operational interruption never becomes an outcome."""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.experiments.plans import freeze_plan, read_plan
from common.io.json import atomic_json, sha
from common.io.records import write_record
from common.runtime.budget import wall_budget
from common.runtime.failures import ProtocolError
from experiments.paper_lifted.evidence.budget import freeze, generation_sources
from experiments.paper_lifted.evidence.design import validation_design
from experiments.paper_lifted.run import calibrated_adapter
from experiments.paper_lifted.study import LiftedStudy
from lifted.runtime.adapters import BudgetExhausted, ContinuationAdapter
from lifted.runtime.records import read, record_path


def fake_pilot(directory):
    plan = freeze_plan(directory / "plan.json", validation_design("pilot"))
    binding = {"backend": "accelerate"}
    atomic_json(directory / "binding.json", binding)
    atomic_json(directory / "analysis/report.json", {"naive_base_equal_pairs": 3})
    for task in plan["tasks"]:
        steps = max(1, task["horizon_steps"]) * len(task["replicates"])
        result = dict(
            states=[],
            paths=[
                dict(replicate=i, step=task["horizon_steps"], category="k2", k_all=2, stop_reason="geometric_terminal")
                for i in task["replicates"]
            ],
            files={},
            binding=binding,
            dependency_sha256=sha(record_path(directory, task["source_task"])) if task["source_task"] else None,
            executed_steps=steps,
            elapsed_seconds=steps * 0.0001,
            wall_seconds=steps * 0.0001 + 0.01,
            source_physical_hash="test",
        )
        write_record(record_path(directory, task["id"]), task, result)
    return plan, binding


def test_freeze_uses_complete_pilot_but_never_builds_or_executes(tmp_path, monkeypatch):
    pilot, destination = tmp_path / "pilot", tmp_path / "formal"
    fake_pilot(pilot)

    def forbidden(*a, **kw):
        raise AssertionError("freeze attempted scientific execution")

    monkeypatch.setattr("common.runtime.go.build_adapter", forbidden)
    monkeypatch.setattr("common.experiments.execution.execute_tasks", forbidden)
    monkeypatch.setattr(ContinuationAdapter, "compute_result", forbidden)
    receipt = freeze(pilot, destination, workers=4, hours=7)
    assert 6 <= receipt["estimate"]["expected_hours"] <= 7
    assert receipt["formal_started"] is False
    assert set(p.name for p in destination.iterdir()) == {"plan.json", "budget.json"}
    saved = read_plan(destination / "plan.json")
    assert saved["profile"] == "formal" and saved["tasks"]
    first = (destination / "plan.json").read_bytes()
    freeze(pilot, destination, workers=4, hours=7)
    assert first == (destination / "plan.json").read_bytes()
    with pytest.raises(ValueError):
        freeze(pilot, destination, workers=4, hours=5)


def test_checkpoint_hash_and_parent_record_are_required(tmp_path):
    task = validation_design("smoke")["tasks"][0]
    checkpoint = tmp_path / "states/state.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"exact-state")
    write_record(
        record_path(tmp_path, task["id"]),
        task,
        dict(files={"states/state.bin": sha(checkpoint)}, binding={}, dependency_sha256=None),
    )
    read(tmp_path, task, {})
    checkpoint.write_bytes(b"changed-state")
    with pytest.raises(ProtocolError, match="checkpoint"):
        read(tmp_path, task, {})


def test_deadline_stops_dispatch_without_creating_scientific_results(tmp_path, monkeypatch):
    plan = freeze_plan(tmp_path / "plan.json", validation_design("smoke"))

    def forbidden(*a, **kw):
        raise AssertionError("expired budget dispatched a task")

    monkeypatch.setattr("experiments.paper_lifted.study.execute_tasks", forbidden)
    study = LiftedStudy(tmp_path, plan, {}, {}, SimpleNamespace(), workers=2, deadline=time.monotonic() - 1)
    result = study.run()
    assert result["missing"] and not (tmp_path / "raw").exists()
    adapter = ContinuationAdapter(tmp_path, {}, {}, {}, SimpleNamespace(), deadline=time.monotonic() - 1)
    with pytest.raises(BudgetExhausted):
        adapter.call({"id": "late"}, Path(tmp_path))


def test_wall_budget_is_shared_by_resume_attempts(tmp_path):
    path = tmp_path / "wall.json"
    tick = [100.0]
    with wall_budget(path, 10, clock=lambda: tick[0], wall=lambda: tick[0]) as deadline:
        assert deadline == 110
        tick[0] = 104
    tick[0] = 200
    with wall_budget(path, 10, clock=lambda: tick[0], wall=lambda: tick[0]) as deadline:
        assert deadline == 206
        tick[0] = 202
    with pytest.raises(ValueError, match="changed"):
        with wall_budget(path, 11):
            pass


def test_cost_revisions_preserve_numerical_pilot_binding():
    original = {
        "experiments/paper_lifted/evidence/budget.py": "before",
        "experiments/paper_lifted/run.py": "before",
        "src/lifted/runtime/costs.py": "before",
        "experiments/paper_lifted/evidence/analyze.py": "before",
        "experiments/paper_lifted/evidence/figures.py": "before",
        "experiments/paper_lifted/__main__.py": "before",
        "src/lifted/runtime/engine/micro.go": "physical",
        "experiments/paper_lifted/evidence/design.py": "design",
    }
    revised = {name: digest if digest != "before" else "after" for name, digest in original.items()}
    assert generation_sources(original) == generation_sources(revised)
    assert generation_sources(original) != generation_sources({**revised, "src/lifted/runtime/engine/micro.go": "new"})
    assert generation_sources(original) != generation_sources(
        {**revised, "experiments/paper_lifted/evidence/design.py": "new"}
    )


def test_formal_stages_verified_pilot_executable_without_rebuilding(tmp_path, monkeypatch):
    pilot, formal = tmp_path / "pilot", tmp_path / "formal"
    source = pilot / "build/lifted"
    source.mkdir(parents=True)
    for name in ("paper-lifted-engine", "go.mod", "go.sum"):
        (source / name).write_text("authenticated " + name)
    (source / "paper-lifted-engine").chmod(0o755)
    manifest = dict(
        binary_sha256=sha(source / "paper-lifted-engine"),
        module_sha256=sha(source / "go.mod"),
        sum_sha256=sha(source / "go.sum"),
        micro={"model.go": "micro"},
        meso={"state.go": "meso"},
        backend="accelerate",
        engine={"main.go": "adapter"},
    )
    atomic_json(source / "manifest.json", manifest)
    calibration = dict(pilot_dir=str(pilot), runtime_binding={"binaries": {"lifted": manifest["binary_sha256"]}})
    monkeypatch.setattr("experiments.paper_lifted.run.engine_manifest", lambda p: manifest["engine"])
    monkeypatch.setattr("experiments.paper_lifted.run.build_adapter", lambda *a, **kw: pytest.fail("unexpected build"))

    def stage(**options):
        return calibrated_adapter(
            formal,
            "lifted",
            tmp_path,
            calibration,
            backend="accelerate",
            micro=options.get("micro", manifest["micro"]),
            meso=manifest["meso"],
        )

    binary = Path(stage())
    assert sha(binary) == manifest["binary_sha256"] and binary.stat().st_mode & 0o111
    assert stage() == str(binary)
    with pytest.raises(ProtocolError, match="source/backend"):
        stage(micro={"model.go": "changed"})
    binary.write_text("tampered")
    with pytest.raises(ProtocolError, match="hash changed"):
        stage()
