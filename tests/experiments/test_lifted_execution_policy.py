"""Execution policy is independent of scientific identity and completion evidence."""

import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from common.experiments.execution import TaskAdapter, execute_tasks
from common.experiments.plans import freeze_plan
from common.experiments.progress import JournalIndex
from common.experiments.receipts import archive_receipt
from common.experiments.scheduling import ReadyTasks
from common.io.json import atomic_json
from common.runtime.budget import extend_wall_budget
from common.runtime.failures import BatchTaskError, ProtocolError, TaskDeferred
from experiments.paper_lifted.__main__ import main, parser
from experiments.paper_lifted.evidence.design import Design, parameters
from experiments.paper_lifted.policy import reference_workers, validate_sources
from experiments.paper_lifted.progress import OverallProgress, inspect_progress
from experiments.paper_lifted.scheduling import fair_key
from experiments.paper_lifted.study import LiftedStudy


def graph():
    d = Design("smoke")
    for ratio in (0.25, 2, 10):
        p = parameters(ratio, n=40, bins=15)
        root = d.add("initial", "initial", p)
        for family in ("micro_probe", "meso_probe"):
            for replica in range(3):
                d.add(family, family, p, parent=root, paths=1, start=replica, steps=1)
    return d.freeze()


def test_ready_stream_obeys_dependencies_and_covers_micro_meso_ratios_first():
    plan = graph()
    done = {t["id"] for t in plan["tasks"] if t["kind"] == "initial"}
    queue = ReadyTasks(plan["tasks"], done, key=fair_key)
    order = []

    def save(task, result):
        assert task["source_task"] in queue.done
        order.append((task["kind"], task["parameters"]["ratio"], min(task["replicates"])))
        queue.complete(task["id"])

    adapter = TaskAdapter(lambda t: t["id"], lambda t: False, lambda t: {}, save, cancel=queue.stop)
    execute_tasks(queue, adapter, workers=1, stream=True, progress=False)
    assert len(order) == 18
    assert len({(kind, ratio) for kind, ratio, _ in order[:6]}) == 6
    assert all(replica == 0 for _, _, replica in order[:6])
    assert len(queue.done) == len(plan["tasks"])


def test_children_become_ready_without_whole_stage_barrier():
    plan = graph()
    queue = ReadyTasks(plan["tasks"], (), key=fair_key)
    done = set()

    def save(task, result):
        assert task["source_task"] is None or task["source_task"] in done
        done.add(task["id"])
        queue.complete(task["id"])

    execute_tasks(
        queue,
        TaskAdapter(lambda t: t["id"], lambda t: False, lambda t: {}, save, cancel=queue.stop),
        workers=3,
        stream=True,
        progress=False,
    )
    assert len(done) == len(plan["tasks"])


def test_cached_failure_cannot_deadlock_waiting_dependency_stream():
    plan = graph()
    queue = ReadyTasks(plan["tasks"], (), key=fair_key)

    def corrupt(task):
        raise ProtocolError("changed input")

    with pytest.raises(ProtocolError):
        execute_tasks(
            queue,
            TaskAdapter(lambda t: t["id"], corrupt, lambda t: {}, lambda t, r: None, cancel=queue.stop),
            workers=4,
            stream=True,
            progress=False,
        )
    assert queue.stopped


def test_deferred_task_is_neither_retried_nor_failed():
    task = graph()["tasks"][0]
    states, calls = [], []
    journal = SimpleNamespace(event=lambda identifier, state, **kw: states.append(state))

    def compute(t):
        calls.append(t["id"])
        raise TaskDeferred("wall budget exhausted")

    execute_tasks(
        [task],
        TaskAdapter(lambda t: t["id"], lambda t: False, compute, lambda t, r: pytest.fail("saved")),
        journal=journal,
        progress=False,
    )
    assert calls == [task["id"]] and states.count("deferred") == 1
    assert not ({"complete", "failed", "retrying"} & set(states))


def test_native_deadline_timeout_becomes_deferred(tmp_path, monkeypatch):
    plan = graph()
    committed, events, calls = set(), [], []

    class Adapter:
        def __init__(self, root, tasks, binaries, binding, store, **kw):
            self.store, self.tasks = store, tasks
            self.cached = self.is_cached

        def is_cached(self, task):
            return task["id"] in committed

        def compute(self, task):
            calls.append(task["id"])
            time.sleep(0.03)
            raise subprocess.TimeoutExpired("native", 0.01)

        def save(self, task, result):
            pytest.fail("cancelled task cannot save a scientific result")

    monkeypatch.setattr("experiments.paper_lifted.study.ContinuationAdapter", Adapter)
    store = SimpleNamespace(read=lambda fn: fn(), event=lambda task, state, **kw: events.append(state))
    study = LiftedStudy(tmp_path, plan, {}, {}, store, workers=1, schedule="balanced", deadline=time.monotonic() + 0.02)
    result = study.run()
    assert result["missing"] and len(calls) == 1
    assert events.count("deferred") == 1 and "failed" not in events and "retrying" not in events
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "partial"


def test_real_task_error_is_still_a_failure(tmp_path, monkeypatch):
    plan = graph()

    class Adapter:
        def __init__(self, root, tasks, binaries, binding, store, **kw):
            self.store = store
            self.cached = self.is_cached

        def is_cached(self, task):
            return False

        def compute(self, task):
            raise ValueError("scientific failure")

        def save(self, *args):
            pytest.fail("saved")

    monkeypatch.setattr("experiments.paper_lifted.study.ContinuationAdapter", Adapter)
    store = SimpleNamespace(read=lambda fn: fn(), event=lambda *a, **kw: None)
    with pytest.raises(BatchTaskError):
        LiftedStudy(tmp_path, plan, {}, {}, store, workers=1, schedule="balanced").run()


def test_explicit_cumulative_budget_extension_preserves_used_time_and_plan(tmp_path):
    ledger = tmp_path / "wall_budget.json"
    atomic_json(ledger, dict(seconds=28800, used=23760, active=False))
    plan = {"budget": {"hard_hours": 8}, "plan_fingerprint": "frozen"}
    atomic_json(tmp_path / "plan.json", plan)
    before = (tmp_path / "plan.json").read_bytes()
    assert extend_wall_budget(ledger, initial_seconds=28800) == 28800
    assert (
        extend_wall_budget(
            ledger, initial_seconds=28800, seconds=72000, metadata={"plan_fingerprint": plan["plan_fingerprint"]}
        )
        == 72000
    )
    assert json.loads(ledger.read_text())["used"] == 23760
    assert (
        extend_wall_budget(
            ledger, initial_seconds=28800, seconds=72000, metadata={"plan_fingerprint": plan["plan_fingerprint"]}
        )
        == 72000
    )
    assert len((tmp_path / "wall_budget_extensions.jsonl").read_text().splitlines()) == 1
    assert (tmp_path / "plan.json").read_bytes() == before
    for invalid in (8, 0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            extend_wall_budget(ledger, initial_seconds=28800, seconds=invalid * 3600)


def test_legacy_worker_reference_is_not_a_scientific_constraint():
    assert reference_workers({"budget": {"workers": 4}}) == 4
    assert reference_workers({"budget": {"reference_workers": 6}}) == 6
    original = {"experiments/paper_lifted/study.py": "old", "src/lifted/runtime/engine/micro.go": "model"}
    current = {**original, "experiments/paper_lifted/study.py": "new", "experiments/paper_lifted/progress.py": "new"}
    assert validate_sources(original, current)
    with pytest.raises(ProtocolError, match="scientific"):
        validate_sources(original, {**current, "src/lifted/runtime/engine/micro.go": "modified"})


def test_freeze_cli_requires_no_workers_and_runtime_policy_is_separate(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("experiments.paper_lifted.evidence.budget.freeze", lambda *a, **kw: seen.append(kw) or {})
    assert main(["freeze", "--work-dir", str(tmp_path), "--pilot-dir", "pilot"]) == 0
    assert seen == [{"workers": None, "hours": 7}]
    a = parser().parse_args(["resume", "--work-dir", str(tmp_path), "--workers", "6", "--wall-hours", "20"])
    assert a.schedule == "balanced" and a.workers == 6 and a.wall_hours == 20


def test_journal_legacy_budget_failure_is_deferred_and_read_only(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(dict(task="x", state="failed", error="wall budget exhausted; task remains pending", time=1)) + "\n"
    )
    before = path.read_bytes()
    index = JournalIndex(path, deferred_errors=("wall budget exhausted",))
    index.refresh()
    assert index.deferred == {"x"} and not index.failed
    index.refresh()
    assert index.deferred == {"x"} and path.read_bytes() == before


def test_progress_unknown_cost_is_not_false_zero_eta(tmp_path):
    plan = freeze_plan(tmp_path / "plan.json", graph())
    progress = OverallProgress(tmp_path, plan)
    snapshot = progress.snapshot()
    assert snapshot["planned"] == 21 and snapshot["completed"] == 0
    assert snapshot["estimated_remaining_seconds"] is None and snapshot["unknown_cost_tasks"] > 0
    assert not (tmp_path / "wall_budget.json").exists()
    assert not (tmp_path / "raw").exists()


def test_eta_scales_with_runtime_workers_and_overrun_is_not_zero(tmp_path, monkeypatch):
    plan = freeze_plan(tmp_path / "plan.json", graph())
    progress = OverallProgress(tmp_path, plan, workers=3)
    progress.costs.predict = lambda task: 10
    a = progress.snapshot()
    assert a["estimated_remaining_seconds"] == pytest.approx(21 * 10 / 3 * 1.15)
    progress.workers = 6
    assert progress.snapshot()["estimated_remaining_seconds"] == pytest.approx(a["estimated_remaining_seconds"] / 2)
    identifier = plan["tasks"][0]["id"]
    progress.index.active[identifier] = time.time() - 20
    late = progress.snapshot()
    assert late["estimated_remaining_seconds"] is None and late["active_estimate_overruns"] == 1


def test_live_workers_and_limit_follow_execution_metadata_without_mutating_plan(tmp_path):
    plan = freeze_plan(tmp_path / "plan.json", graph())
    atomic_json(tmp_path / "execution.json", {"workers": 6, "core_reserve_seconds": 2400})
    atomic_json(tmp_path / "wall_budget.json", {"seconds": 28800, "used": 23760, "active": False})
    progress = OverallProgress(tmp_path, plan)
    value = progress.snapshot(refresh=True)
    assert value["workers"] == 6 and value["core_dispatch_remaining_seconds"] == 2640


def test_short_validation_limit_can_be_created_and_persists(tmp_path):
    assert extend_wall_budget(tmp_path / "wall_budget.json", initial_seconds=0, seconds=36) == 36
    assert extend_wall_budget(tmp_path / "wall_budget.json", initial_seconds=28800) == 36


def test_monitor_includes_frozen_followup_without_scanning_its_large_inputs(tmp_path):
    d = Design("smoke")
    p = parameters(n=40, bins=15)
    initial = d.add("initial", "initial", p)
    d.add("queries", "micro_probe", p, parent=initial, paths=1, steps=1)
    plan = freeze_plan(tmp_path / "plan.json", d.freeze())
    f = Design("smoke")
    f.add("queries", "meso_probe", p, parent=initial, paths=1, steps=1)
    freeze_plan(
        tmp_path / "followup/plan.json",
        {
            **f.freeze(),
            "parent_plan_fingerprint": plan["plan_fingerprint"],
            "inputs": {"raw/large-unavailable-data.msgpack.lz4": "not-read-by-operational-monitor"},
        },
    )
    assert inspect_progress(tmp_path)["planned"] == 3
    assert not (tmp_path / "wall_budget.json").exists()


def test_old_attempt_receipt_is_archived_before_last_attempt_changes(tmp_path):
    atomic_json(tmp_path / "attempt.json", {"failure": "wall budget exhausted", "missing": ["task"]})
    original = (tmp_path / "attempt.json").read_bytes()
    archive_receipt(tmp_path / "attempt.json", tmp_path / "executions")
    archive_receipt(tmp_path / "attempt.json", tmp_path / "executions")
    files = list((tmp_path / "executions").glob("receipt-*.json"))
    assert len(files) == 1 and files[0].read_bytes() == original
    assert (tmp_path / "attempt.json").read_bytes() == original
