"""The offline runner finishes downstream work only after complete collection."""

import json
import subprocess
from dataclasses import replace
from ehk.domain import to_wire
from tests.experiments import numerical_fixtures as f
from experiments.paper.metric_comparison.contracts import PathStatus,PipelineStatus,PipelinePaths

import pytest

from common.experiments.lifecycle import session_lock
from experiments.paper.metric_comparison import run as runner


@pytest.fixture(autouse=True)
def forbid_native_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline pipeline tests must not start native processes")

    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _complete():
    return f.collection()


def _failed(*errors,pending=0):
    paths=tuple(PathStatus("staged",str(index),"late","hold","plan","binary","raw", "failed",error=error)
        for index,error in enumerate(errors))
    return f.collection(status="failed",planned_paths=len(errors)+pending,completed_paths=0,
        pending_paths=pending,failed_paths=len(errors),paths=paths)


def _stages(monkeypatch, *, collection=None):
    calls = []

    def collect(preparation, output, **options):
        calls.append(("collect", preparation, output, options))
        return _complete() if collection is None else collection()

    def compare(preparation, collection, output):
        calls.append(("alignment", preparation, collection, output))
        output.mkdir()
        return f.comparison()

    def report(preparation, comparison, output, **options):
        calls.append(("report", preparation, comparison, output, options))
        output.mkdir()
        return f.report()

    monkeypatch.setattr(runner.raw, "collect_raw", collect)
    monkeypatch.setattr(runner.core, "compare", compare)
    monkeypatch.setattr(runner.report, "report", report)
    return calls


@pytest.mark.parametrize("reuse", [False, True])
def test_complete_collection_automatically_aligns_and_reports(tmp_path, monkeypatch, reuse):
    calls = _stages(monkeypatch)
    preparation, output = tmp_path / "preparation", tmp_path / "pipeline"
    collection = tmp_path / "existing-collection" if reuse else None
    selected = collection if reuse else output / "collection"
    result = runner.run(preparation, output, collection=collection, workers=8, make_figures=False)
    assert calls == [
        ("collect", preparation, selected, {"workers": 8, "endpoint_complement": True}),
        ("alignment", preparation, selected, output / "alignment"),
        ("report", preparation, output / "alignment", output / "report", {"make_figures": False}),
    ]
    assert result.state == "complete" and result.phase == "report"
    assert result.collection.completed_paths == 2
    assert result.paths.collection == str(selected)
    assert runner.core.load_contract(output / "run_status.json",PipelineStatus) == result
    assert len(result.runner_sha256) == 64


def test_only_source_change_failures_retry_the_same_collection_configuration(tmp_path, monkeypatch):
    results = iter([_failed(runner.RETRYABLE_SOURCE_CHANGE), _complete()])
    calls = _stages(monkeypatch, collection=lambda: next(results))
    output = tmp_path / "pipeline"
    result = runner.run(tmp_path / "preparation", output, workers=3)
    assert [call[0] for call in calls] == ["collect", "collect", "alignment", "report"]
    assert calls[0] == calls[1]
    assert result.attempt == 2 and result.state == "complete"
    assert result.reason is None


@pytest.mark.parametrize("failure", [
    _failed("unreadable raw record"),
    _failed(runner.RETRYABLE_SOURCE_CHANGE, "different confidence radius"),
    _failed(runner.RETRYABLE_SOURCE_CHANGE, pending=1),
    f.collection(status="pending",completed_paths=1,pending_paths=1),
    replace(_complete(),completed_paths=1,pending_paths=1),
])
def test_failed_or_missing_data_stop_without_zero_filling_or_downstream_work(tmp_path, monkeypatch, failure):
    calls = _stages(monkeypatch, collection=lambda: failure)
    output = tmp_path / "pipeline"
    result = runner.run(tmp_path / "preparation", output)
    assert [call[0] for call in calls] == ["collect"]
    assert result.state != "complete" and result.phase == "collect"
    assert result.collection.completed_paths == failure.completed_paths
    assert not (output / "alignment").exists() and not (output / "report").exists()


def test_retry_attempts_are_bounded(tmp_path, monkeypatch):
    calls = _stages(monkeypatch, collection=lambda: _failed(runner.RETRYABLE_SOURCE_CHANGE))
    result = runner.run(tmp_path / "preparation", tmp_path / "pipeline", attempts=2)
    assert [call[0] for call in calls] == ["collect", "collect"]
    assert result.state == "failed" and result.attempt == 2


@pytest.mark.parametrize("name", ["alignment", "report"])
def test_existing_alignment_or_report_is_never_overwritten(tmp_path, monkeypatch, name):
    calls = _stages(monkeypatch)
    output = tmp_path / "pipeline"
    target = output / name
    target.mkdir(parents=True)
    sentinel = target / "original"
    sentinel.write_bytes(b"preserve these results")
    with pytest.raises(ValueError, match="already exists"):
        runner.run(tmp_path / "preparation", output)
    assert calls == [] and sentinel.read_bytes() == b"preserve these results"
    saved = json.loads((output / "run_status.json").read_text())
    assert saved["state"] == "failed" and saved["phase"] == "preflight"


def test_interruption_is_visible_and_resumes_existing_path_receipts(tmp_path, monkeypatch):
    preparation, output = tmp_path / "preparation", tmp_path / "pipeline"
    collection = tmp_path / "existing-collection"
    receipt = collection / "records" / "completed.json.gz"

    def interrupted():
        receipt.parent.mkdir(parents=True)
        receipt.write_bytes(b"committed path receipt")
        raise KeyboardInterrupt("stop after committing one path")

    calls = _stages(monkeypatch, collection=interrupted)
    with pytest.raises(KeyboardInterrupt, match="committing"):
        runner.run(preparation, output, collection=collection)
    saved = json.loads((output / "run_status.json").read_text())
    assert saved["state"] == "interrupted" and saved["phase"] == "collect"
    assert receipt.read_bytes() == b"committed path receipt"
    assert [call[0] for call in calls] == ["collect"]

    def resumed():
        assert receipt.read_bytes() == b"committed path receipt"
        return _complete()

    calls = _stages(monkeypatch, collection=resumed)
    assert runner.run(preparation, output, collection=collection).state == "complete"
    assert [call[0] for call in calls] == ["collect", "alignment", "report"]


def test_exception_is_visible_and_is_not_automatically_retried(tmp_path, monkeypatch):
    def corrupt():
        raise ValueError("frozen metric configuration changed")

    calls = _stages(monkeypatch, collection=corrupt)
    output = tmp_path / "pipeline"
    with pytest.raises(ValueError, match="configuration changed"):
        runner.run(tmp_path / "preparation", output)
    assert [call[0] for call in calls] == ["collect"]
    assert json.loads((output / "run_status.json").read_text())["state"] == "failed"


def test_pending_alignment_never_starts_the_report(tmp_path, monkeypatch):
    calls = _stages(monkeypatch)
    monkeypatch.setattr(runner.core, "compare", lambda *args: f.comparison(status="pending_metrics"))
    result = runner.run(tmp_path / "preparation", tmp_path / "pipeline")
    assert [call[0] for call in calls] == ["collect"]
    assert result.state == "pending" and result.phase == "alignment"


def test_pipeline_lock_does_not_overwrite_an_active_run_status(tmp_path, monkeypatch):
    calls = _stages(monkeypatch)
    output = tmp_path / "pipeline"
    with session_lock(output / "runtime"):
        (output / "run_status.json").write_text('{"state":"running"}')
        with pytest.raises(RuntimeError, match="already active"):
            runner.run(tmp_path / "preparation", output)
    assert calls == []
    assert json.loads((output / "run_status.json").read_text()) == {"state": "running"}


@pytest.mark.parametrize("field,value", [("workers", 0), ("workers", True), ("attempts", 0), ("attempts", 1.5)])
def test_invalid_worker_and_attempt_limits_do_not_start_collection(tmp_path, monkeypatch, field, value):
    calls = _stages(monkeypatch)
    with pytest.raises(ValueError, match="positive integers"):
        runner.run(tmp_path / "preparation", tmp_path / "pipeline", **{field: value})
    assert calls == []


def test_cli_exposes_complete_pipeline_and_nonzero_pending_exit(tmp_path, monkeypatch, capsys):
    calls = []

    def pending(*args, **kwargs):
        calls.append((args, kwargs))
        return PipelineStatus(schema=1,state="pending",phase="collect",attempt=0,workers=8,attempts=2,make_figures=False,endpoint_complement=True,runner_sha256="s",paths=PipelinePaths("p","o","c","a","r","g"))

    monkeypatch.setattr(runner, "run", pending)
    preparation, output, collection = (tmp_path / name for name in ("prepared", "pipeline", "collection"))
    code = runner.main(["--preparation", str(preparation), "--output", str(output), "--collection", str(collection),
                        "--workers", "8", "--attempts", "2", "--no-figures"])
    assert calls == [((preparation, output), {"collection": collection, "workers": 8, "attempts": 2,
                                            "make_figures": False, "endpoint_complement": True})]
    assert code == 1
    assert json.loads(capsys.readouterr().out)["state"] == "pending"
