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
