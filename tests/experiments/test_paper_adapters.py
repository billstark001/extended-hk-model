"""Paper analysis documents and shared simulation adapters."""

import numpy as np

from common.experiments.execution import execute_tasks
from experiments.paper.analysis.records import ScenarioStatistics, StatisticsStore
from experiments.paper.analysis.statistics import analyze_statistics


def fake_statistics(metadata, base, origin, previous):
    if metadata.get("defer"):
        return None
    if metadata["UniqueName"] == "failed":
        raise ValueError("intentional fixture failure")
    return ScenarioStatistics(
        name=metadata["UniqueName"],
        origin=origin,
        tolerance=0.4,
        decay=0.02,
        rewiring=0.04,
        repost_rate=0,
        recsys_type="Random",
        post_retain_count=3,
        step=(previous.step + 1) if previous else 1,
        x_indices=np.array([1, 2], dtype=np.int32),
    )


def deferred_statistics(metadata, base, origin, previous):
    return None


def test_statistics_process_pool_upserts_documents_and_keeps_origins(tmp_path):
    database = tmp_path / "statistics.sqlite"
    scenarios = [{"UniqueName": "existing"}, {"UniqueName": "new"}, {"UniqueName": "failed"}]
    result = analyze_statistics(
        fake_statistics,
        str(tmp_path),
        database,
        "original",
        scenarios,
        mode="fixture",
        ignore_exist=False,
        concurrency=2,
    )
    assert result["missing"] == ["failed"] and result["state"] == "partial"
    with StatisticsStore(database, readonly=True) as store:
        rows = store.records()
    assert len(rows) == 2
    assert rows[0].x_indices.dtype == np.int32
    assert rows[0].metadata["UniqueName"] == rows[0].name
    assert rows[0].protocol["metrics"]["revision"] == 3
    assert rows[0].source["identity_method"]
    again = analyze_statistics(
        fake_statistics,
        str(tmp_path),
        database,
        "original",
        scenarios,
        mode="fixture",
        ignore_exist=True,
        concurrency=1,
    )
    assert [r["state"] for r in again["records"] if r["id"] != "failed"] == ["cached", "cached"]
    analyze_statistics(fake_statistics, str(tmp_path), database, "other", scenarios[:1], mode="fixture")
    with StatisticsStore(database, readonly=True) as store:
        assert len(store.records()) == 3
        assert store.get("existing", "original", result["analysis_id"]).step == 1


def test_protocol_metadata_and_raw_changes_invalidate_completed_results(tmp_path):
    database = tmp_path / "statistics.sqlite"
    metadata = {"UniqueName": "existing", "RNG": {"Seed1": "7"}}
    kwargs = dict(mode="fixture", ignore_exist=False, is_complete=lambda row: row.complete)
    first = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [metadata], **kwargs)
    second = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [metadata], **kwargs)
    assert second["records"][0]["state"] == "cached"
    raw = tmp_path / "existing"
    raw.mkdir()
    (raw / "trajectory.json").write_text('{"channels":1}')
    changed = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [metadata], **kwargs)
    assert changed["records"][0]["state"] == "complete"
    new_protocol = analyze_statistics(
        fake_statistics, str(tmp_path), database, "raw", [metadata], protocol={"example": "changed"}, **kwargs
    )
    assert new_protocol["analysis_id"] != first["analysis_id"]
    with StatisticsStore(database, readonly=True) as store:
        assert len(store.records()) == 2
    forced = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [metadata], force=True, **kwargs)
    assert forced["records"][0]["state"] == "complete"
    with StatisticsStore(database, readonly=True) as store:
        assert store.get("existing", "raw", first["analysis_id"]).step == 2


def test_failed_reanalysis_invalidates_old_results_and_subset_can_expand(tmp_path):
    database = tmp_path / "statistics.sqlite"
    first_task = {"UniqueName": "existing"}
    first = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [first_task], mode="fixture")
    full = analyze_statistics(
        fake_statistics, str(tmp_path), database, "raw", [first_task, {"UniqueName": "second"}], mode="fixture"
    )
    assert full["completed"] == 2 and first["analysis_id"] == full["analysis_id"]
    deferred = {**first_task, "defer": True}
    result = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [deferred], mode="fixture")
    assert result["missing"] == ["existing"]
    with StatisticsStore(database, readonly=True) as store:
        record = store.get("existing", "raw", first["analysis_id"])
        assert record.stale and not record.complete
        assert "changed" in record.stale_reason
    retry = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [deferred], mode="fixture")
    assert retry["records"][0]["state"] == "deferred"


def test_force_failure_cannot_be_reused_by_default_resume(tmp_path):
    database = tmp_path / "statistics.sqlite"
    tasks = [{"UniqueName": "existing"}]
    initial = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", tasks, mode="fixture")
    failed = analyze_statistics(deferred_statistics, str(tmp_path), database, "raw", tasks, mode="fixture", force=True)
    assert failed["records"][0]["state"] == "deferred"
    resumed = analyze_statistics(deferred_statistics, str(tmp_path), database, "raw", tasks, mode="fixture")
    assert resumed["records"][0]["state"] == "deferred"
    with StatisticsStore(database, readonly=True) as store:
        assert store.get("existing", "raw", initial["analysis_id"]).stale


def test_smp_adapter_preserves_metadata_and_batch_finished_markers(tmp_path, monkeypatch):
    import smp_bindings.simulation as binding

    from ehk.micro.simulation import SimulationAdapter

    tasks = [{"id": str(i), "metadata": {"UniqueName": str(i), "RNGSeed": [3, 7]}} for i in range(2)]
    finished = set()

    def single(binary, workspace, metadata, show_progress, print_lock, **kwargs):
        metadata["RNGSeed"][0] = 99  # The public binding may normalize metadata in place.
        finished.add(metadata["UniqueName"])
        return metadata["UniqueName"]

    def batch(binary, workspace, scenarios, **kwargs):
        assert kwargs == {"max_concurrent": 2, "show_position": True, "skip_finished": False}
        return [single(binary, workspace, m, False, None) for m in scenarios]

    monkeypatch.setattr(binding, "run_simulation", single)
    monkeypatch.setattr(binding, "run_simulations", batch)
    monkeypatch.setattr(binding, "is_simulation_finished", lambda workspace, m: m["UniqueName"] in finished)
    monkeypatch.setattr("ehk.micro.simulation.run_isolated", lambda f, *args: f(*args))
    execute_tasks(tasks, SimulationAdapter(tmp_path / "binary", tmp_path), workers=2)
    assert finished == {"0", "1"}
    finished.clear()
    execute_tasks(tasks, SimulationAdapter(tmp_path / "binary", tmp_path, binding_batch=True), workers=2)
    assert finished == {"0", "1"}
    assert all(t["metadata"]["RNGSeed"] == [3, 7] for t in tasks)
