"""Model-independent experiment lifecycle and cancellation contracts."""

import io
import json
from pathlib import Path

import pytest

from common.experiments.execution import TaskAdapter, execute_tasks
from common.experiments.lifecycle import TaskJournal, Workflow
from common.experiments.plans import coverage, freeze_plan, read_plan
from common.experiments.progress import Progress
from common.experiments.registry import TaskRegistry
from common.io.json import atomic_json
from common.runtime.failures import BatchTaskError, ProtocolError
from common.runtime.process import run_isolated


def test_registry_aggregates_and_frozen_identity_reject_incompatible_seeds(tmp_path):
    registry = TaskRegistry({"one": 1, "two": 2}, {"all": ("one", "two")})
    assert registry.select("all") == (1, 2)
    with pytest.raises(ValueError):
        registry.select("alias")
    candidate = {"schema": 1, "tasks": [{"id": "a", "seed": 3}]}
    path = tmp_path / "plan.json"
    saved = freeze_plan(path, candidate)
    original = path.read_bytes()
    assert freeze_plan(path, candidate) == saved
    with pytest.raises(ProtocolError, match="plan changed"):
        freeze_plan(path, {**candidate, "tasks": [{"id": "a", "seed": 4}]})
    assert path.read_bytes() == original
    forged = {**saved, "tasks": [{"id": "a", "seed": 4}]}
    atomic_json(path, forged)
    with pytest.raises(ProtocolError, match="hash mismatch"):
        read_plan(path)
    assert coverage(candidate["tasks"], []) == {"planned": 1, "completed": 0, "missing": ["a"]}


def test_task_owned_batch_reports_partial_success_and_resume_skips_committed(tmp_path):
    tasks = [{"id": str(i)} for i in range(3)]
    saved, calls = {}, []

    def batch(pending, workers, complete, failed):
        calls.append([t["id"] for t in pending])
        for task in pending:
            if task["id"] == "1" and len(calls) == 1:
                failed(task, RuntimeError("scientific failure"))
            else:
                complete(task, task["id"])

    adapter = TaskAdapter(
        lambda t: t["id"],
        lambda t: t["id"] in saved,
        lambda t: None,
        lambda t, r: saved.update({t["id"]: r}),
        batch=batch,
    )
    journal = TaskJournal(tmp_path)
    with pytest.raises(BatchTaskError):
        execute_tasks(tasks, adapter, journal=journal)
    assert saved == {"0": "0", "2": "2"}
    assert journal.finish(tasks)["state"] == "partial"
    execute_tasks(tasks, adapter, journal=journal)
    assert calls == [["0", "1", "2"], ["1"]]
    assert journal.finish(tasks)["missing"] == []


def test_fancy_progress_tracks_duration_and_does_not_estimate_from_cache():
    stream = io.StringIO()
    tick = [0.0]
    progress = Progress(3, stream=stream, clock=lambda: tick[0])
    progress.event("cached", "cached")
    progress.event("run", "running")
    tick[0] = 2.5
    progress.event("run", "complete")
    output = stream.getvalue()
    assert "[1/3]" in output and "ETA —" in output
    assert "[2/3]" in output and "2.5s" in output and "ETA 2.5s" in output


def test_common_package_does_not_recognize_model_bindings():
    import ast

    import common.experiments

    root = Path(common.experiments.__file__).parent
    for source in root.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("smp_", "theory", "ehk", "experiments"))
            elif isinstance(node, ast.Import):
                assert all(not n.name.startswith(("smp_", "theory", "ehk", "experiments")) for n in node.names)


def test_workflow_records_failure_and_stops_later_stages(tmp_path):
    seen = []

    def failure():
        raise ValueError("failed stage")

    workflow = Workflow({"run": failure, "plot": lambda: seen.append("plot")}, tmp_path / "attempt.json")
    with pytest.raises(ValueError):
        workflow.sequence(("run", "plot"))
    assert seen == []
    assert json.loads((tmp_path / "attempt.json").read_text())["state"] == "failed"


def test_batch_persistence_failure_is_isolated_and_workers_are_validated():
    tasks = [{"id": "broken"}, {"id": "saved"}]
    saved = []

    def persist(task, result):
        if task["id"] == "broken":
            raise ValueError("cannot persist")
        saved.append(result)

    def batch(tasks, workers, complete, failed):
        for task in tasks:
            complete(task, task["id"])

    adapter = TaskAdapter(lambda t: t["id"], lambda t: False, lambda t: None, persist, batch=batch)
    with pytest.raises(ValueError, match="positive workers"):
        execute_tasks(tasks, adapter, workers=0)
    with pytest.raises(BatchTaskError):
        execute_tasks(tasks, adapter)
    assert saved == ["saved"]


def isolated_fixture(value):
    if value < 0:
        raise ValueError("isolated failure")
    return {"value": value}


def isolated_descendant(marker):
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    Path(marker).write_text(str(child.pid))
    child.wait()


def test_opaque_native_launcher_preserves_results_errors_and_cancels_descendants(tmp_path):
    import os
    import signal
    import subprocess
    import threading
    import time

    assert run_isolated(isolated_fixture, 3) == {"value": 3}
    with pytest.raises(ValueError, match="isolated failure"):
        run_isolated(isolated_fixture, -1)
    marker = tmp_path / "pid"

    def interrupt_when_started():
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Thread(target=interrupt_when_started, daemon=True)
    timer.start()
    with pytest.raises(KeyboardInterrupt):
        run_isolated(isolated_descendant, str(marker))
    timer.join(timeout=1)
    assert marker.is_file()
    child_pid = int(marker.read_text())
    # A killed, adopted child can briefly remain a zombie; it must not be running.
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True)
    assert not result.stdout.strip() or result.stdout.strip().startswith("Z")
