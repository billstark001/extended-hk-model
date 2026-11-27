"""Paper analysis documents and shared simulation adapters."""

import numpy as np
from dataclasses import replace
from ehk.domain import Rng
from ehk.micro.scenarios import create_scenario_metadata
from experiments.paper.analysis.models import AnalysisOptions

from common.experiments.execution import execute_tasks
from ehk.metrics.protocol import METRIC_REVISION
from experiments.paper.analysis.records import ScenarioStatistics, StatisticsStore
from experiments.paper.analysis.statistics import analyze_statistics


def fake_statistics(metadata, base, origin, previous):
    if metadata.MaxSimulationStep == 1:
        return None
    if metadata.UniqueName == "failed":
        raise ValueError("intentional fixture failure")
    return ScenarioStatistics(
        name=metadata.UniqueName,
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
    scenarios = [create_scenario_metadata(name) for name in ("existing", "new", "failed")]
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
    assert result.missing == ("failed",) and result.state == "partial"
    with StatisticsStore(database, readonly=True) as store:
        rows = store.records()
    assert len(rows) == 2
    assert rows[0].x_indices.dtype == np.int32
    assert rows[0].metadata.UniqueName == rows[0].name
    assert rows[0].protocol.metrics.revision == METRIC_REVISION
    assert rows[0].source.identity_method
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
    assert [r.state for r in again.records if r.id != "failed"] == ["cached", "cached"]
    analyze_statistics(fake_statistics, str(tmp_path), database, "other", scenarios[:1], mode="fixture")
    with StatisticsStore(database, readonly=True) as store:
        assert len(store.records()) == 3
        assert store.get("existing", "original", result.analysis_id).step == 1


def test_protocol_metadata_and_raw_changes_invalidate_completed_results(tmp_path):
    database = tmp_path / "statistics.sqlite"
    metadata = replace(create_scenario_metadata("existing"), RNG=Rng(Algorithm="pcg64-dxsm-v1", Seed1="7", Seed2="9"))
    kwargs = dict(mode="fixture", ignore_exist=False, is_complete=lambda row: row.complete)
    first = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [metadata], **kwargs)
    second = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [metadata], **kwargs)
    assert second.records[0].state == "cached"
    raw = tmp_path / "existing"
    raw.mkdir()
    (raw / "trajectory.json").write_text('{"channels":1}')
    changed = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [metadata], **kwargs)
    assert changed.records[0].state == "complete"
    new_protocol = analyze_statistics(
        fake_statistics, str(tmp_path), database, "raw", [metadata], protocol=AnalysisOptions(active_threshold=.99), **kwargs
    )
    assert new_protocol.analysis_id != first.analysis_id
    with StatisticsStore(database, readonly=True) as store:
        assert len(store.records()) == 2
    forced = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [metadata], force=True, **kwargs)
    assert forced.records[0].state == "complete"
    with StatisticsStore(database, readonly=True) as store:
        assert store.get("existing", "raw", first.analysis_id).step == 2


def test_failed_reanalysis_invalidates_old_results_and_subset_can_expand(tmp_path):
    database = tmp_path / "statistics.sqlite"
    first_task = create_scenario_metadata("existing")
    first = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [first_task], mode="fixture")
    full = analyze_statistics(
        fake_statistics, str(tmp_path), database, "raw", [first_task, create_scenario_metadata("second")], mode="fixture"
    )
    assert full.completed == 2 and first.analysis_id == full.analysis_id
    deferred = replace(first_task, MaxSimulationStep=1)
    result = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [deferred], mode="fixture")
    assert result.missing == ("existing",)
    with StatisticsStore(database, readonly=True) as store:
        record = store.get("existing", "raw", first.analysis_id)
        assert record.stale and not record.complete
        assert "changed" in record.stale_reason
    retry = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", [deferred], mode="fixture")
    assert retry.records[0].state == "deferred"


def test_force_failure_cannot_be_reused_by_default_resume(tmp_path):
    database = tmp_path / "statistics.sqlite"
    tasks = [create_scenario_metadata("existing")]
    initial = analyze_statistics(fake_statistics, str(tmp_path), database, "raw", tasks, mode="fixture")
    failed = analyze_statistics(deferred_statistics, str(tmp_path), database, "raw", tasks, mode="fixture", force=True)
    assert failed.records[0].state == "deferred"
    resumed = analyze_statistics(deferred_statistics, str(tmp_path), database, "raw", tasks, mode="fixture")
    assert resumed.records[0].state == "deferred"
    with StatisticsStore(database, readonly=True) as store:
        assert store.get("existing", "raw", initial.analysis_id).stale
