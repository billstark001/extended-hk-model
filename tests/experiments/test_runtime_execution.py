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


def test_selected_workflow_help_does_not_create_outputs(tmp_path, monkeypatch):
    dispatched = []
    monkeypatch.setattr(cli, "module_call", lambda module, args: dispatched.append((module, args)))
    output = tmp_path / "absent"
    cli.main(["scan", "--workflow", "baseline", "--output-root", str(output), "--help"])
    assert dispatched[0][0].endswith("baseline.cli")
    assert "--help" in dispatched[0][1]
    assert not output.exists()


@pytest.mark.parametrize("workflow", ["baseline", "expanded", "scaling"])
def test_scan_dispatch_only_passes_supported_worker_option(tmp_path, monkeypatch, workflow):
    dispatched = []
    monkeypatch.setattr(cli, "module_call", lambda module, args: dispatched.append((module, args)))
    cli.main(["scan", "--workflow", workflow, "--output-root", str(tmp_path), "--analysis-only"])
    module, args = dispatched[0]
    assert module == f"experiments.paper_theory.energy_scan.{workflow}.cli"
    assert ("--workers" in args) == (workflow == "expanded")
    assert "--analysis-only" in args


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


@pytest.mark.parametrize(
    "exit_code, stages", [(1, ["build", "plan", "run", "analyze"]), (130, ["build", "plan", "run"])]
)
def test_pipeline_analyzes_failures_but_stops_on_interrupt(tmp_path, monkeypatch, exit_code, stages):
    seen = []

    def perform(args, extra):
        seen.append(args.action)
        if args.action == "run":
            raise SystemExit(exit_code)

    monkeypatch.setattr(cli, "perform", perform)
    with pytest.raises(SystemExit) as error:
        cli.main(["pipeline", "--suite", "validation", "--profile", "smoke", "--output-root", str(tmp_path)])
    assert error.value.code == exit_code
    assert seen == stages
