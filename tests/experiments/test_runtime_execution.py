"""Execution contracts that protect resumable scientific runs."""

import json
import math
import sys
import time
from pathlib import Path

import pytest

from common.io.json import atomic_json, fingerprint, read_result
from common.runtime.environment import runtime_environment
from experiments.paper_theory import __main__ as cli
from experiments.paper_theory.runtime.paths import OutputPaths
from theory.runtime.dataset import checked_records, validate
from theory.runtime.executor import STOP, execute
from theory.runtime.protocol import prepare_task, task_line
from theory.runtime.store import write_design


def executable(tmp_path, body):
    path = tmp_path / "fake-engine"
    path.write_text(f"#!{sys.executable}\nimport sys, json, time\n" + body)
    path.chmod(0o755)
    return path


def job():
    return {
        "id": "test",
        "family": "micro",
        "tier": "core",
        "meta": {},
        "payload": {"id": "test", "kind": "micro", "process_seed": 42},
    }


def test_transport_sets_explicit_policies_without_mutating_frozen_tasks():
    frozen = job()["payload"]
    original = json.loads(json.dumps(frozen))
    prepared = prepare_task(frozen)
    assert prepared["initial_rng"] == "shared"
    assert prepared["projection"] == "fixed_degree_rows"
    assert json.loads(task_line(frozen)) == prepared
    assert frozen == original


def test_subprocess_cache_paths_survive_a_changed_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    environment = runtime_environment(Path("outputs/run"))
    for key in ("TMPDIR", "GOTMPDIR", "GOCACHE", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX"):
        path = Path(environment[key])
        assert path.is_absolute() and path.is_dir()


def test_resume_cannot_overwrite_a_frozen_design(tmp_path):
    path = tmp_path / "design.json"
    tasks = [job()]
    assert write_design(path, iter(tasks)) == tasks
    original = path.read_bytes()
    assert write_design(path, tasks) == tasks
    with pytest.raises(ValueError, match="design changed"):
        write_design(path, [{**job(), "tier": "extended"}])
    assert path.read_bytes() == original


def test_removed_cli_profiles_and_workflows_create_no_outputs(tmp_path):
    output = tmp_path / "absent"
    for args in (["scan"], ["clocks"], ["plan", "--profile", "paper"], ["run", "--suite", "validation"]):
        with pytest.raises(SystemExit) as error:
            cli.main([*args, "--output-dir", str(output)])
        assert error.value.code == 2
    assert not output.exists()


def test_csv_plot_dispatch_never_imports_a_solver(tmp_path, monkeypatch):
    from experiments.paper_theory.evidence import table_figures

    (tmp_path / "plan.json").write_text("{}")
    seen = []
    monkeypatch.setattr(table_figures, "plot_tables", lambda directory, output: seen.append(directory))
    cli.main(["plot", "--run-dir", str(tmp_path), "--tables-only"])
    assert seen == [tmp_path]


@pytest.mark.parametrize("action", ["run", "resume"])
def test_generation_dispatch_uses_one_parser_and_structured_arguments(tmp_path, monkeypatch, action):
    from experiments.paper_theory.evidence import run_main

    seen = []
    monkeypatch.setattr(run_main, "launch", lambda args: seen.append(args))
    path_option = "--run-dir" if action == "resume" else "--work-dir"
    cli.main([action, path_option, str(tmp_path / "bulk"), "--publish-dir", str(tmp_path / "export")])
    assert len(seen) == 1
    assert seen[0].work_dir == tmp_path / "bulk"
    assert seen[0].publish_dir == tmp_path / "export"
    assert seen[0].resume == (action == "resume")
    assert not (tmp_path / "bulk").exists()


def test_execute_resume_and_protocol_errors(tmp_path):
    STOP.clear()
    outputs = OutputPaths(tmp_path / "outputs")
    binary = executable(
        tmp_path,
        "task=json.loads(sys.stdin.readline())\nprint(json.dumps({'id':task['id'],'status':'ok','result':{'value':3}}))\n",
    )
    current = job()
    result = execute(current, binary, "source", "smoke_test", math.inf, 0, outputs)
    assert result["status"] == "complete"
    saved = read_result(Path(result["path"]))
    assert saved["result"] == {"value": 3}
    assert saved["job"] == current
    assert execute(current, binary, "source", "smoke_test", math.inf, 0, outputs)["status"] == "cached"
    with pytest.raises(RuntimeError, match="incompatible result"):
        execute(current, binary, "changed-source", "smoke_test", math.inf, 0, outputs)
    binary = executable(
        tmp_path,
        "task=json.loads(sys.stdin.readline())\nprint(json.dumps({'id':task['id'],'status':'error','error':'bad request'}))\n",
    )
    failed = execute(current, binary, "source", "smoke_error", math.inf, 0, outputs)
    assert failed["status"] == "failed" and "bad request" in failed["stderr"]
    assert not (outputs.family("micro", "smoke_error") / "raw" / "test.json.gz").exists()


def test_deadlines_are_distinct_from_task_timeouts(tmp_path):
    STOP.clear()
    outputs = OutputPaths(tmp_path / "outputs")
    binary = executable(tmp_path, "time.sleep(10)\n")
    assert (
        execute(job(), binary, "source", "smoke_deadline", time.monotonic() - 1, 0, outputs)["status"]
        == "budget_unstarted"
    )
    assert execute(job(), binary, "source", "smoke_timeout", math.inf, 0.05, outputs)["status"] == "job_timeout"


def test_bound_readers_reject_forged_job_and_corrupted_plan(tmp_path):
    STOP.clear()
    outputs = OutputPaths(tmp_path / "outputs")
    root = outputs.control("smoke_test")
    current = job()
    plan = {"schema": 2, "jobs": [current]}
    plan_id = fingerprint(plan)
    atomic_json(root / "plan.json", {**plan, "plan_fingerprint": plan_id})
    atomic_json(root / "binding.json", {"plan_fingerprint": plan_id, "source_fingerprint": "source"})
    _, binding, expected = validate(root)
    binary = executable(
        tmp_path,
        "task=json.loads(sys.stdin.readline())\nprint(json.dumps({'id':task['id'],'status':'ok','result':{}}))\n",
    )
    execute(current, binary, "source", "smoke_test", math.inf, 0, outputs)
    assert len(checked_records(outputs, "smoke_test", "micro", binding, expected)) == 1
    atomic_json(root / "plan.json", {**plan, "jobs": [], "plan_fingerprint": plan_id})
    with pytest.raises(ValueError, match="content/hash mismatch"):
        validate(root)
