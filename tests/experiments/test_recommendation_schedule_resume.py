"""Rescheduling counts existing results before executing unfinished tasks."""

import json

import pytest

from common.io.json import fingerprint, save_result
from experiments.paper.recommendation import study


def cached_fixture(tmp_path, monkeypatch, *, pending):
    binary = tmp_path / "smp"
    binary.write_bytes(b"binary fingerprint fixture")
    cached = {"id": "later-repeat", "replicate": 8, "metadata": {"UniqueName": "later-repeat"}}
    unfinished = {"id": "first-repeat", "replicate": 0, "metadata": {"UniqueName": "first-repeat"}}
    tasks = [unfinished, cached] if pending else [cached]
    monkeypatch.setattr(study.cfg, "generate_tasks", lambda profile: tasks)
    monkeypatch.setattr(study, "_raw_complete", lambda directory, arm: True)
    directory = tmp_path / "dataset"
    save_result(directory / "records" / "later-repeat.json.gz", {
        "task_fingerprint": fingerprint(cached), "result": {"arms": {"hold": {"raw_record": "raw/cached"}}},
    })
    return directory, binary, tasks


def test_interrupt_retains_cached_later_repeats_in_completion_status(tmp_path, monkeypatch):
    directory, binary, tasks = cached_fixture(tmp_path, monkeypatch, pending=True)

    def interrupted(pending, adapter, *, journal, **kwargs):
        assert list(pending) == [tasks[0]]
        assert journal.records[tasks[1]["id"]]["state"] == "cached"
        raise KeyboardInterrupt("reschedule fixture")

    monkeypatch.setattr(study, "execute_tasks", interrupted)
    with pytest.raises(KeyboardInterrupt, match="reschedule fixture"):
        study.run_study("recommender_variants", directory, binary, 1)
    status = json.loads((directory / "runtime" / "status.json").read_text())
    assert status["state"] == "interrupted" and status["completed"] == 1
    assert status["missing"] == [tasks[0]["id"]]


def test_all_cached_tasks_complete_without_starting_a_simulator(tmp_path, monkeypatch):
    directory, binary, _ = cached_fixture(tmp_path, monkeypatch, pending=False)

    def unexpected_run(*args):
        pytest.fail("rescheduling restarted a completed simulation")

    monkeypatch.setattr(study, "run_task", unexpected_run)
    status = study.run_study("recommender_variants", directory, binary, 1)
    assert status["completed"] == status["planned"] == 1 and status["missing"] == []
    assert status["records"][0]["state"] == "cached"
