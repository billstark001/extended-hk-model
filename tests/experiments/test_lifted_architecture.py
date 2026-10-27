"""Owning-layer contracts and shared execution primitives after the audit."""

import ast
import json
from pathlib import Path

import pytest

from common.experiments.execution import TaskAdapter, execute_ready_tasks
from common.experiments.lifecycle import StageJournals
from common.runtime.failures import ProtocolError
from experiments.paper_lifted.evidence.design import Design, parameters
from lifted.runtime.forecast import ContinuationCosts


def test_new_shared_services_have_no_model_or_study_dependencies():
    root = Path(__file__).resolve().parents[2]
    for name in ("execution", "progress", "scheduling", "receipts", "lifecycle"):
        tree = ast.parse((root / "src/common/experiments" / (name + ".py")).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"lifted", "theory", "ehk", "experiments"}
    tree = ast.parse((root / "src/lifted/runtime/forecast.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name not in {"pathlib", "json"} for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("experiments", "common.io", "pathlib"))


def test_deferred_persistence_does_not_release_a_child():
    tasks = [dict(id="parent", source_task=None), dict(id="child", source_task="parent")]
    computed, snapshots = [], []
    adapter = TaskAdapter(lambda t: t["id"], lambda t: False, lambda t: computed.append(t["id"]), lambda t, r: False)
    done = execute_ready_tasks(
        tasks, adapter, completed=(), key=lambda t: ("one", t["id"]), progress=False, on_finish=snapshots.append
    )
    assert computed == ["parent"] and done == set() and snapshots == [set()]


def test_protocol_failure_cancels_stream_and_publishes_committed_coverage():
    tasks = [dict(id="bad", source_task=None), dict(id="child", source_task="bad")]
    cancelled, snapshots = [], []

    def cached(task):
        raise ProtocolError("changed input")

    adapter = TaskAdapter(
        lambda t: t["id"], cached, lambda t: None, lambda t, r: None, cancel=lambda: cancelled.append(True)
    )
    with pytest.raises(ProtocolError):
        execute_ready_tasks(
            tasks,
            adapter,
            completed=(),
            key=lambda t: ("one", t["id"]),
            workers=2,
            progress=False,
            on_finish=snapshots.append,
        )
    assert cancelled and snapshots == [set()]


def test_stage_coverage_includes_previously_committed_tasks(tmp_path):
    journals = StageJournals(tmp_path, [dict(name="micro", task_ids=["a", "b"]), dict(name="empty", task_ids=[])])
    journals.publish({"a"})
    report = json.loads((tmp_path / "micro/coverage.json").read_text())
    assert report == dict(planned=2, completed=1, missing=["b"], state="partial")
    assert json.loads((tmp_path / "empty/coverage.json").read_text())["state"] == "complete"


def test_pure_forecast_uses_supplied_native_rate_and_mother_future():
    d = Design("smoke")
    p = parameters(n=500, bins=15)
    atlas = d.add("atlas", "micro_atlas", p, steps=6000, observe=(6000,))
    future = d.add("truth", "micro_tail", p, parent=atlas, at=6000, steps=24000)
    query = d.add("query", "micro_probe", p, parent=atlas, at=6000, paths=8, steps=24000)
    tasks = {t["id"]: t for t in d.tasks.values()}
    metadata = {
        atlas: dict(states=[dict(step=6000, category="unresolved")], paths=[{}]),
        future: dict(paths=[dict(step=10000, category="k1")]),
    }
    costs = ContinuationCosts(tasks, metadata=metadata.get)
    costs.rates[500].append(0.003)
    assert costs.predict(tasks[query]) == pytest.approx(96.02)
    metadata[atlas]["states"][0]["category"] = "k1"
    assert costs.predict(tasks[query]) == 0.1
