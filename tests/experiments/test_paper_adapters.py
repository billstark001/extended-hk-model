"""Original-paper SMP and SQLite adapter compatibility."""

import json

import numpy as np
from sqlalchemy import select

from common.experiments.execution import execute_tasks
from common.storage.sqlite import create_db_engine_and_session
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.statistics import analyze_statistics


def fake_statistics(metadata, base, origin, previous):
    if metadata["UniqueName"] == "failed":
        raise ValueError("intentional fixture failure")
    return ScenarioStatistics(
        id=previous.id if previous else None,
        name=metadata["UniqueName"],
        origin=origin,
        tolerance=0.4,
        decay=0.02,
        rewiring=0.04,
        retweet=0,
        recsys_type="Random",
        tweet_retain_count=3,
        step=(previous.step + 1) if previous else 1,
        x_indices=np.array([1, 2], dtype=np.int32),
    )


def test_statistics_process_pool_updates_existing_sqlite_rows_and_keeps_origins(tmp_path):
    database = tmp_path / "stats.db"
    scenarios = [{"UniqueName": "existing"}, {"UniqueName": "new"}, {"UniqueName": "failed"}]
    engine, session = create_db_engine_and_session(str(database), ScenarioStatistics.Base)
    session.add(fake_statistics(scenarios[0], "", "original", None))
    session.add(fake_statistics(scenarios[0], "", "other", None))
    session.commit()
    original_id = session.scalars(select(ScenarioStatistics).where(ScenarioStatistics.origin == "original")).one().id
    session.close()
    engine.dispose()
    analyze_statistics(
        fake_statistics,
        str(tmp_path),
        str(database),
        "original",
        scenarios,
        mode="fixture",
        ignore_exist=False,
        concurrency=2,
    )
    engine, session = create_db_engine_and_session(str(database), ScenarioStatistics.Base)
    rows = list(session.scalars(select(ScenarioStatistics)))
    assert len(rows) == 3
    previous = next(r for r in rows if r.origin == "original" and r.name == "existing")
    assert previous.id == original_id and previous.step == 2
    assert previous.x_indices.dtype == np.int32
    assert next(r for r in rows if r.origin == "other").step == 1
    session.close()
    engine.dispose()
    report = json.loads(next((tmp_path / "runtime").glob("*/status.json")).read_text())
    assert report["missing"] == ["failed"] and report["state"] == "partial"
    analyze_statistics(
        fake_statistics,
        str(tmp_path),
        str(database),
        "original",
        scenarios,
        mode="fixture",
        ignore_exist=True,
        concurrency=1,
    )
    engine, session = create_db_engine_and_session(str(database), ScenarioStatistics.Base)
    assert session.get(ScenarioStatistics, original_id).step == 2
    session.close()
    engine.dispose()


def test_statistics_study_completion_predicate_avoids_deferred_existing_results(tmp_path):
    database = tmp_path / "stats.db"
    metadata = {"UniqueName": "existing"}
    engine, session = create_db_engine_and_session(str(database), ScenarioStatistics.Base)
    session.add(fake_statistics(metadata, "", "original", None))
    session.commit()
    session.close()
    engine.dispose()
    result = analyze_statistics(
        fake_statistics,
        str(tmp_path),
        str(database),
        "original",
        [metadata],
        mode="fixture",
        ignore_exist=False,
        is_complete=lambda previous: previous.step == 1,
    )
    assert result["missing"] == [] and result["completed"] == 1
    assert result["records"][0]["state"] == "cached"


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
