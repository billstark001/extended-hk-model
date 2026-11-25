"""Checks for frozen inputs and storage-independent correctness measurements."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.io.json import fingerprint
from experiments.storage_benchmark import corpus, worker
from experiments.storage_benchmark import run as driver
from experiments.storage_benchmark.common import document_digest, header_document, payload_size
from experiments.storage_benchmark.run import run


def test_content_check_covers_array_shape_dtype_and_nested_values():
    array = np.arange(8, dtype=np.int32)
    original = {"_id": "a", "payload": {"nested": [array]}, "complete": True}
    assert document_digest(original) == document_digest(dict(reversed(list(original.items()))))
    for changed in (array.reshape(2, 4), array.astype(np.float32), array + 1):
        assert document_digest(original) != document_digest({**original, "payload": {"nested": [changed]}})
    assert payload_size(original["payload"]) == 32
    assert header_document(original) == {"_id": "a", "complete": True}


def test_changed_frozen_plan_fails_before_worker_launch(tmp_path):
    plan = {"study": "isolated_storage_benchmark", "repetitions": 3}
    plan["fingerprint"] = fingerprint(plan)
    plan["repetitions"] = 1
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan changed"):
        run(tmp_path)


def test_changed_catalog_fails_before_worker_launch(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("changed")
    plan = {
        "study": "isolated_storage_benchmark",
        "source_files": {},
        "profiles": {"sample": {"catalog": str(catalog), "catalog_sha256": "stale"}},
    }
    plan["fingerprint"] = fingerprint(plan)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="catalog changed"):
        run(tmp_path)


@pytest.mark.parametrize("backend", ("current", "tinymongo", "indexed_split"))
def test_worker_updates_then_genuinely_replaces_full_documents(tmp_path, monkeypatch, backend):
    if backend == "tinymongo":
        pytest.importorskip("tinymongo")
    documents = [
        {"_id": str(index), "complete": True, "stale": False, "payload": {"array": np.arange(8) + index}}
        for index in range(2)
    ]
    monkeypatch.setattr(worker, "get_profile", lambda _: {"count": len(documents)})
    monkeypatch.setattr(worker, "iter_documents", lambda *_args, **_kwargs: iter(deepcopy(documents)))
    monkeypatch.setattr(worker, "document_at", lambda index, _profile: deepcopy(documents[index]))
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "".join(
            json.dumps(
                {
                    "_id": document["_id"],
                    "index": index,
                    "full_digest": document_digest(document),
                    "header_digest": document_digest(header_document(document)),
                }
            )
            + "\n"
            for index, document in enumerate(documents)
        )
    )
    request = {"profile": "fixture", "backend": backend, "directory": str(tmp_path / "db"), "catalog": str(catalog)}
    for action in ("load", "state_update", "replace_full"):
        result = worker.run({**request, "action": action})
        assert result["status"] == "ok"
        assert result["latency"]["operations"] == 2
    with worker.open_backend(backend, tmp_path / "db", readonly=True) as store:
        for index, expected in enumerate(documents):
            expected.update(complete=False, stale=True, benchmark_replacement_epoch=1)
            assert document_digest(store.get(str(index))) == document_digest(expected)


def _roundtrip_fixture(tmp_path, monkeypatch, *, sorted_fields):
    sources = {
        "lengths": np.array([3, 5], dtype=np.int64),
        "length_probabilities": np.array([0.5, 0.5]),
        "actual_files": [tmp_path / f"source-{index}" for index in range(3)],
        "tasks": [
            {"mode": family}
            for family in ("epsilon", "epsilon", "gradation", "gradation", "mech", "mech", "replicate", "replicate")
        ],
        "probe_shapes": {"z": [3], "a": [3]},
    }

    def loader():
        return sources

    monkeypatch.setattr(driver, "load_sources", loader)
    monkeypatch.setattr(corpus, "load_sources", loader)

    def generate(index, profile):
        state = corpus.load_sources()
        assert state["lengths"].dtype == np.int64
        assert all(isinstance(path, Path) for path in state["actual_files"])
        revision, task = divmod(index, len(state["tasks"]))
        family = state["tasks"][task]["mode"]
        rng = np.random.default_rng(index)
        names = sorted(state["probe_shapes"]) if sorted_fields else state["probe_shapes"]
        payload = {name: rng.normal(size=3) for name in names} if family == "mech" else {"series": rng.normal(size=3)}
        return {"_id": f"{profile}-{index}", "mode": family, "analysis_id": str(revision), "payload": payload}

    monkeypatch.setattr(corpus, "document_at", generate)
    return sources, loader


def test_preflight_covers_families_all_mechanisms_and_revision_boundaries(tmp_path, monkeypatch):
    sources, loader = _roundtrip_fixture(tmp_path, monkeypatch, sorted_fields=True)
    packed = driver._sources_pack(tmp_path)
    path, report = driver._preflight_source_pack(tmp_path, packed, driver.PROFILES)
    assert corpus.load_sources is loader
    assert report["status"] == "passed" and path.is_file()
    by_profile = report["profiles"]
    assert {sample["family"] for sample in by_profile["full_paper"]} == {task["mode"] for task in sources["tasks"]}
    selected = {sample["index"] for sample in by_profile["versioned_2x"]}
    assert {4, 5, 12, 13, 7, 8, 9} <= selected
    assert len(by_profile["actual_available"]) == 3
    assert driver._unpack_sources(packed)["actual_files"] == sources["actual_files"]


def test_preflight_rejects_rng_order_changes_before_even_limited_catalog_generation(tmp_path, monkeypatch):
    _sources, loader = _roundtrip_fixture(tmp_path, monkeypatch, sorted_fields=False)
    monkeypatch.setattr(
        driver, "iter_documents", lambda *_args, **_kwargs: pytest.fail("catalog generated before preflight")
    )
    with pytest.raises(ValueError, match="index=4 family=mech"):
        driver.prepare(tmp_path / "new-campaign", tmp_path / "vendor", profiles=("full_paper",), limit=1)
    assert corpus.load_sources is loader
    report = json.loads((tmp_path / "new-campaign/source-pack-preflight.json").read_text())
    assert report["status"] == "failed"
    assert not (tmp_path / "new-campaign/plan.json").exists()
    assert not list((tmp_path / "new-campaign").glob("catalog-*.jsonl"))


@pytest.fixture
def prepare_fixture(tmp_path, monkeypatch):
    _roundtrip_fixture(tmp_path, monkeypatch, sorted_fields=True)
    monkeypatch.setattr(driver, "get_profile", lambda _profile: {"count": 2})
    monkeypatch.setattr(
        driver,
        "iter_documents",
        lambda profile, limit=None: (corpus.document_at(index, profile) for index in range(limit or 2)),
    )
    monkeypatch.setattr(driver, "_hardware", lambda: {"fixture": True})
    return tmp_path / "vendor"


def test_prepare_keeps_original_backend_and_action_defaults(tmp_path, prepare_fixture):
    plan = driver.prepare(tmp_path / "campaign", prepare_fixture, profiles=("full_paper",))
    assert plan["backends"] == ["current", "tinymongo", "indexed_split"]
    assert plan["actions"] == [
        "point_metadata",
        "point_full",
        "selective_metadata",
        "scan_metadata",
        "scan_full",
        "nested_metadata",
        "state_update",
        "replace_full",
    ]
    assert plan["source_pack_preflight"]["checked_documents"] > 0
    frozen = {key: value for key, value in plan.items() if key != "fingerprint"}
    assert fingerprint(frozen) == plan["fingerprint"]


def test_prepare_freezes_subset_order_and_only_selected_recovery_links(tmp_path, prepare_fixture):
    (tmp_path / "recovery_final").mkdir()
    plan = driver.prepare(
        tmp_path / "campaign",
        prepare_fixture,
        profiles=("full_paper",),
        backends=("sqlite_optimized", "indexed_split"),
        actions=("replace_full", "scan_full", "state_update"),
    )
    assert plan["backends"] == ["sqlite_optimized", "indexed_split"]
    assert plan["actions"] == ["scan_full", "state_update", "replace_full"]
    assert set(plan["recovery"]) == {"sqlite_optimized", "indexed_split"}
    frozen = {key: value for key, value in plan.items() if key != "fingerprint"}
    for field, replacement in (("backends", ["indexed_split"]), ("actions", ["scan_full"])):
        assert fingerprint({**frozen, field: replacement}) != plan["fingerprint"]


@pytest.mark.parametrize(
    "selection",
    [
        {"backends": ()},
        {"backends": None},
        {"backends": "indexed_split"},
        {"backends": ("missing",)},
        {"backends": ("indexed_split", "indexed_split")},
        {"actions": ()},
        {"actions": None},
        {"actions": ("load",)},
        {"actions": ("missing",)},
        {"actions": ("scan_full", "scan_full")},
    ],
)
def test_invalid_selections_fail_before_preparation(tmp_path, monkeypatch, selection):
    monkeypatch.setattr(driver, "_sources_pack", lambda _directory: pytest.fail("loaded sources for invalid selection"))
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="backends|actions"):
        driver.prepare(directory, tmp_path / "vendor", **selection)
    assert not directory.exists()


def test_run_rotates_only_selected_backends_from_the_frozen_plan(tmp_path, prepare_fixture, monkeypatch):
    plan = driver.prepare(
        tmp_path / "campaign",
        prepare_fixture,
        profiles=("actual_available", "full_paper"),
        backends=("sqlite_optimized", "indexed_split"),
        actions=("scan_full",),
        repetitions=2,
    )
    monkeypatch.syspath_prepend(plan["vendor"])
    captured = []
    monkeypatch.setattr(driver, "_run_jobs", lambda _directory, jobs, _environment: captured.extend(jobs))
    driver.run(tmp_path / "campaign")
    assert len(captured) == 16
    # JSON plan keys are sorted on disk: actual_available, then full_paper.
    for offset, expected in (
        (0, ["sqlite_optimized", "indexed_split"]),
        (2, ["indexed_split", "sqlite_optimized"]),
        (8, ["indexed_split", "sqlite_optimized"]),
    ):
        assert [request["backend"] for _, request, _ in captured[offset : offset + 2]] == expected
    assert {request["action"] for _, request, _ in captured} == {"load", "scan_full"}
    assert {request["backend"] for _, request, _ in captured} == {"indexed_split", "sqlite_optimized"}


@pytest.mark.parametrize("field,value", [("backends", ["indexed_split"]), ("actions", ["scan_full"])])
def test_changed_frozen_selection_fails_before_worker_launch(tmp_path, prepare_fixture, field, value):
    directory = tmp_path / "campaign"
    plan = driver.prepare(directory, prepare_fixture, profiles=("full_paper",))
    plan[field] = value
    (directory / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan changed"):
        driver.run(directory)


@pytest.mark.parametrize("field,value", [("backends", []), ("backends", ["missing"]), ("actions", [])])
def test_invalid_frozen_selection_is_rejected_before_worker_launch(tmp_path, prepare_fixture, field, value):
    directory = tmp_path / "campaign"
    plan = driver.prepare(directory, prepare_fixture, profiles=("full_paper",))
    plan[field] = value
    plan.pop("fingerprint")
    plan["fingerprint"] = fingerprint(plan)
    (directory / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="backends|actions"):
        driver.run(directory)


def test_subset_resume_skips_only_matching_successful_jobs(tmp_path, prepare_fixture, monkeypatch):
    directory = tmp_path / "campaign"
    plan = driver.prepare(
        directory,
        prepare_fixture,
        profiles=("full_paper",),
        repetitions=1,
        backends=("indexed_split", "sqlite_optimized"),
        actions=("scan_full",),
    )
    monkeypatch.syspath_prepend(plan["vendor"])
    monkeypatch.setattr(driver, "_observe_environment", lambda *_args: None)
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda _directory: SimpleNamespace(free=100 * 1024**3))
    calls = []

    def execute(request, result_path, *_args):
        calls.append(request)
        result = {"status": "ok", "request_fingerprint": fingerprint(request)}
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(driver, "_execute", execute)
    assert driver.run(directory) == {"completed": 4, "planned": 4}
    assert driver.run(directory) == {"completed": 4, "planned": 4}
    assert len(calls) == 4
    assert {request["backend"] for request in calls} == {"indexed_split", "sqlite_optimized"}
    assert {request["action"] for request in calls} == {"load", "scan_full"}


def test_prepare_cli_passes_explicit_selections_and_retains_defaults(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "prepare", lambda *_args, **kwargs: calls.append(kwargs))
    base = ["prepare", "--directory", str(tmp_path)]
    driver.main(base)
    assert calls[-1]["backends"] == driver.BACKENDS
    assert calls[-1]["actions"] == driver.ACTIONS
    driver.main([*base, "--backends", "indexed_split", "sqlite_optimized", "--actions", "scan_metadata", "scan_full"])
    assert calls[-1]["backends"] == ["indexed_split", "sqlite_optimized"]
    assert calls[-1]["actions"] == ["scan_metadata", "scan_full"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["prepare", "--backends"],
        ["prepare", "--backends", "missing"],
        ["prepare", "--actions"],
        ["prepare", "--actions", "missing"],
        ["run", "--backends", "sqlite_optimized"],
        ["report", "--actions", "scan_full"],
    ],
)
def test_cli_rejects_missing_unknown_or_unfrozen_overrides(tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        driver.main([*arguments, "--directory", str(tmp_path)])
    assert error.value.code == 2


@pytest.mark.parametrize("error_type,state", [(RuntimeError, "failed"), (KeyboardInterrupt, "interrupted")])
def test_worker_failure_updates_driver_status_and_observes_outside_worker(tmp_path, monkeypatch, error_type, state):
    observations = []
    monkeypatch.setattr(
        driver, "_observe_environment", lambda _directory, phase, _identifier: observations.append(phase)
    )
    monkeypatch.setattr(driver.shutil, "disk_usage", lambda _directory: SimpleNamespace(free=100 * 1024**3))

    def fail(*_args, **_kwargs):
        raise error_type("fixture failure")

    monkeypatch.setattr(driver, "_execute", fail)
    request = {"action": "point_metadata", "directory": str(tmp_path / "db")}
    with pytest.raises(error_type, match="fixture failure"):
        driver._run_jobs(tmp_path, [("fixture", request, tmp_path / "results/fixture.json")], {})
    result = json.loads((tmp_path / "status.json").read_text())
    assert result["state"] == state and result["completed"] == 0 and result["failed_job"] == "fixture"
    assert observations == ["before", "after_failed"]


def test_environment_observation_records_names_without_arguments(tmp_path, monkeypatch):
    def ps(command, **_kwargs):
        assert command == ["ps", "-axo", "pid=,ppid=,pcpu=,rss=,comm="]
        return SimpleNamespace(
            stdout="123 1 85.1 1200 /Applications/Example.app/Contents/MacOS/Example\n456 1 0.2 300 /usr/bin/python\n"
        )

    monkeypatch.setattr(driver.subprocess, "run", ps)
    driver._observe_environment(tmp_path, "before", "fixture")
    observed = json.loads((tmp_path / "environment-observations.jsonl").read_text())
    assert observed["processes"][0]["command"] == "Example"
    assert "args" not in observed["processes"][0]
