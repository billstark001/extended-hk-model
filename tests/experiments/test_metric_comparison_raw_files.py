"""Saved-file inventory follows real graph names and excludes SQLite lock state."""

import pandas as pd
import pytest

from ehk.domain import freeze_plan
from dataclasses import replace
from tests.experiments import recommendation_fixtures as fixtures
from common.io.json import atomic_json, fingerprint, save_result, sha
from experiments.paper.metric_comparison import core, raw
from experiments.paper.recommendation.design import generate_tasks
from experiments.paper.recommendation_shared.storage import StudyStore


def _raw_files(directory, graph_name="graph-0.msgpack.lz4"):
    directory.mkdir(parents=True)
    for name in ("metadata.json", "trajectory.json", "events.db", "observations.json", "finished.txt", graph_name):
        (directory / name).write_bytes(name.encode())


@pytest.mark.parametrize("name", ["graph-0.msgpack", "graph-0.msgpack.lz4", "graph-000000.msgpack", "graph-000000.msgpack.lz4"])
def test_graph_zero_inventory_accepts_reader_supported_names(tmp_path, name):
    _raw_files(tmp_path / "raw", name)
    assert raw._required_files(tmp_path / "raw") == ()


def test_graph_sidecars_backups_and_directories_do_not_replace_initial_snapshot(tmp_path):
    directory = tmp_path / "raw"
    _raw_files(directory, "graph-16.msgpack.lz4")
    for name in ("._graph-0.msgpack.lz4", "graph-0.msgpack.lz4.bak", "graph-zero.msgpack"):
        (directory / name).touch()
    (directory / "graph-0.msgpack").mkdir()
    assert raw._required_files(directory) == ("graph-0.msgpack[.lz4]",)
    assert set(raw._raw_sources(directory)) == {
        "metadata.json", "trajectory.json", "events.db", "graph-16.msgpack.lz4",
    }


def test_source_hashes_bind_all_graph_snapshots_and_wal_but_ignore_shm(tmp_path):
    _raw_files(tmp_path / "raw")
    directory = tmp_path / "raw"
    for name in ("graph-16.msgpack", "graph-000128.msgpack.lz4", "events.db-wal", "events.db-shm"):
        (directory / name).write_bytes(name.encode())
    before = raw._raw_sources(directory)
    assert {"graph-0.msgpack.lz4", "graph-16.msgpack", "graph-000128.msgpack.lz4", "events.db-wal"} <= before.keys()
    assert "events.db-shm" not in before
    (directory / "events.db-shm").write_bytes(b"updated reader locks")
    assert raw._raw_sources(directory) == before
    (directory / "graph-16.msgpack").write_bytes(b"changed graph")
    after_graph = raw._raw_sources(directory)
    assert after_graph["graph-16.msgpack"] != before["graph-16.msgpack"]
    (directory / "events.db-wal").write_bytes(b"changed events")
    assert raw._raw_sources(directory)["events.db-wal"] != after_graph["events.db-wal"]


@pytest.mark.parametrize("graph_name", ["graph-0.msgpack.lz4", "graph-000000.msgpack"])
def test_preparation_uses_same_graph_inventory_as_raw_collection(tmp_path, graph_name):
    task = generate_tasks("staged_timing")[0]
    data = tmp_path / "bundle" / "data"
    dataset = tmp_path / "disk"
    raw_record = f"raw/{task.id}/full/hold"
    _raw_files(dataset / raw_record, graph_name)
    plan = freeze_plan(data / "study/plan.json",fixtures.plan("staged_timing",tasks=(task,)))
    with StudyStore(data / "study") as store:
        from ehk.domain import FullResult
        store.put_task(task,FullResult(arms={"hold":fixtures.arm(endpoints=(fixtures.point(),),raw_record=raw_record)}),plan=plan)
    atomic_json(data / "study/runtime/binding.json", {"binary_sha256": "saved-binary"})
    manifest=replace(fixtures.bundle_manifest(completed=1,planned=1,profile=plan.profile,plan_fingerprint=plan.plan_fingerprint),
        source_dataset=str(dataset),files={path.relative_to(data).as_posix():sha(path) for path in data.rglob("*") if path.is_file()})
    atomic_json(data / "manifest.json",manifest)
    output = tmp_path / "preparation"
    report = core.prepare([data.parent], output)
    assert report.available_raw_paths == 1
    assert report.pending_raw_paths == 0
    assert pd.read_csv(output / "paths.csv").iloc[0]["available"]

def test_source_identity_binds_the_metadata_and_history_decoders():
    import ehk.domain
    import smp_bindings.domain
    import smp_bindings.events_db
    import smp_bindings.model_state
    import smp_bindings.trajectory
    from pathlib import Path
    source=raw._source_configuration()
    for module in (ehk.domain,smp_bindings.domain,smp_bindings.events_db,smp_bindings.model_state,smp_bindings.trajectory):
        path=Path(module.__file__).resolve()
        assert source.source_sha256[str(path)]==sha(path)
