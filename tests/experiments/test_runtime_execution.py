"""Execution contracts that protect resumable scientific runs."""

import json
import sys
from pathlib import Path

import pytest

from common.experiments.plans import write_design
from common.runtime.environment import runtime_environment
from experiments.paper_theory import __main__ as cli
from theory.runtime.protocol import prepare_task, task_line


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
    from experiments.paper_theory.evidence import figures

    (tmp_path / "plan.json").write_text("{}")
    seen = []
    monkeypatch.setattr(figures, "plot", lambda directory, output: seen.append((directory, output)))
    cli.main(["plot", "--run-dir", str(tmp_path), "--tables-only"])
    assert seen == [(tmp_path, tmp_path / "figures_csv")]


@pytest.mark.parametrize("action", ["run", "resume"])
def test_generation_dispatch_uses_one_parser_and_structured_arguments(tmp_path, monkeypatch, action):
    from experiments.paper_theory import run as run_main

    seen = []
    monkeypatch.setattr(run_main, "launch", lambda args: seen.append(args))
    path_option = "--run-dir" if action == "resume" else "--work-dir"
    cli.main([action, path_option, str(tmp_path / "bulk"), "--publish-dir", str(tmp_path / "export")])
    assert len(seen) == 1
    assert seen[0].work_dir == tmp_path / "bulk"
    assert seen[0].publish_dir == tmp_path / "export"
    assert seen[0].resume == (action == "resume")
    assert not (tmp_path / "bulk").exists()
