"""Rescheduling counts existing results before executing unfinished tasks."""

import json

import pytest

from dataclasses import replace

from ehk.domain import FullResult, Task, freeze_plan
from tests.experiments.recommendation_fixtures import arm
from ehk.micro.scenarios import create_scenario_metadata
from experiments.paper.recommendation_shared.storage import StudyStore
from experiments.paper.recommendation_shared import runtime
from experiments.paper.recommender_variants import study


def cached_fixture(tmp_path, monkeypatch, *, pending):
    binary = tmp_path / "smp"
    binary.write_bytes(b"binary fingerprint fixture")
    cached = Task(id="later-repeat", replicate=8, metadata=create_scenario_metadata("later-repeat"))
    unfinished = Task(id="first-repeat", replicate=0, metadata=create_scenario_metadata("first-repeat"))
    tasks = [unfinished, cached] if pending else [cached]
    monkeypatch.setattr(study.cfg, "generate_tasks", lambda: tasks)
    monkeypatch.setattr(runtime, "complete_raw", lambda directory, arm: True)
    directory = tmp_path / "dataset"
    plan = freeze_plan(directory / "plan.json", study.make_plan())
    with StudyStore(directory) as store:
        store.put_task(cached, FullResult(arms={name: replace(arm(), raw_record="raw/cached") for name in study.cfg.variant_arms()}), plan=plan)
    return directory, binary, tasks


def test_interrupt_retains_cached_later_repeats_in_completion_status(tmp_path, monkeypatch):
    directory, binary, tasks = cached_fixture(tmp_path, monkeypatch, pending=True)

    def interrupted(pending, adapter, *, journal, **kwargs):
        assert list(pending) == [tasks[0]]
        assert journal.records[tasks[1].id].state == "cached"
        raise KeyboardInterrupt("reschedule fixture")

    monkeypatch.setattr(runtime, "execute_tasks", interrupted)
    with pytest.raises(KeyboardInterrupt, match="reschedule fixture"):
        study.run_study(directory, binary, 1)
    status = json.loads((directory / "runtime" / "status.json").read_text())
    assert status["state"] == "interrupted" and status["completed"] == 1
    assert status["missing"] == [tasks[0].id]


def test_all_cached_tasks_complete_without_starting_a_simulator(tmp_path, monkeypatch):
    directory, binary, _ = cached_fixture(tmp_path, monkeypatch, pending=False)

    def unexpected_run(*args):
        pytest.fail("rescheduling restarted a completed simulation")

    monkeypatch.setattr(study, "run_task", unexpected_run)
    status = study.run_study(directory, binary, 1)
    assert status.completed == status.planned == 1 and status.missing == ()
    assert status.records[0].state == "cached"
