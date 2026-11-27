"""Compare legacy replay, native frozen measurements, and persistent exposure traces."""

import json
import sqlite3
import struct
import subprocess
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import lz4.frame
import msgpack
import numpy as np
import pytest
from smp_bindings import RawSimulationRecord
from smp_bindings.migrate import migrate_events_db
from smp_bindings.probe import run_probe_checkpoints

from ehk.micro.traces import iter_exposure_trace
from ehk.micro.persistent import observe_persistent
from smp_bindings.domain import PolicyChange, RecommenderParams, Rng, parse_metadata, to_wire
from ehk.micro.observables import followee_post_concordance, get_pathway_passage
from ehk.micro.scenarios import create_scenario_metadata, reproducible_metadata


@pytest.fixture(scope="module")
def persistent_binary(tmp_path_factory):
    binary = tmp_path_factory.mktemp("smp") / "smp"
    source = Path(__file__).resolve().parents[3] / "social-media-models"
    subprocess.run(["go", "build", "-o", str(binary), "./cmd/smp"], cwd=source, check=True)
    return binary


@pytest.fixture(scope="module")
def probe_binary(tmp_path_factory):
    binary = tmp_path_factory.mktemp("smp-probe") / "smp-probe"
    source = Path(__file__).resolve().parents[3] / "social-media-models"
    subprocess.run(["go", "build", "-o", str(binary), "./cmd/smp-probe"], cwd=source, check=True)
    return binary


@pytest.mark.parametrize("kind", ["counts", "ViewPosts", "ViewTweets"])
def test_legacy_observations_remain_replayable(tmp_path, probe_binary, kind):
    run = tmp_path / "legacy"
    run.mkdir()
    metadata = {
        "UniqueName": "legacy",
        "HKParams": {"Tolerance": 0.45, "RewiringRate": 0.0},
        "AgentNumber": kind == "counts",
        "ViewPostsEvent": kind in ("ViewPosts", "ViewTweets"),
    }
    (run / "metadata.json").write_text(json.dumps(metadata))
    (run / "finished").touch()

    opinions = np.array([[0, 0, 0, 0], [-0.7, -0.7, 0.7, 0.7]], dtype="<f4")
    counts = np.zeros((2, 4, 4), dtype="<i2")
    if kind == "counts":
        counts[1, :, :] = [1, 1, 1, 0]
    accum = struct.pack("<ii", 2, 4) + opinions.tobytes() + counts.tobytes() + bytes(2 * 4 * 4 * 4)
    (run / "acc-state-000001.lz4").write_bytes(lz4.frame.compress(accum))
    graph = {"directed": True, "nodes": {i: {} for i in range(4)}, "adjacency": {i: {i ^ 1: {}} for i in range(4)}}
    (run / "graph-000000.msgpack").write_bytes(msgpack.packb(graph))
    with closing(sqlite3.connect(run / "events.db")) as db, db:
        db.executescript("""
            CREATE TABLE events(id INTEGER PRIMARY KEY, type TEXT, agent_id INTEGER, step INTEGER);
            CREATE TABLE rewiring_events(event_id INTEGER PRIMARY KEY, unfollow INTEGER, follow INTEGER);
            CREATE TABLE view_posts_events(event_id INTEGER PRIMARY KEY, data BLOB);
            CREATE TABLE view_tweets_events(event_id INTEGER PRIMARY KEY, data BLOB);
        """)
        if kind != "counts":
            db.executemany("INSERT INTO events VALUES (?, ?, ?, 1)", [(i + 1, kind, i) for i in range(4)])
            table = "view_posts_events" if kind == "ViewPosts" else "view_tweets_events"
            post = {"AgentID": 2, "Step": 0, "Opinion": 0.3}
            body = {
                "NeighborConcordant": [(0, 0, 0.1)],
                "NeighborDiscordant": [post],
                "RecommendedConcordant": [(1, 0, 0.2)],
                "RecommendedDiscordant": [],
            }
            db.executemany(f"INSERT INTO {table} VALUES (?, ?)", [(i + 1, msgpack.packb(body)) for i in range(4)])

    migrate_events_db(str(run / "events.db"))
    with RawSimulationRecord(str(tmp_path), parse_metadata(metadata)) as record:
        assert get_pathway_passage(record, probe_binary) == ("simultaneous", 1, 1)
        assert np.isnan(followee_post_concordance(record, 0))
        assert followee_post_concordance(record, 1) == 0.5
        assert record.feed_counts(1).sum(axis=0).tolist() == [4, 4, 4, 0]
        if kind != "counts":
            events = list(record.iter_feed_events(1, 2))
            assert all(event.type == "ViewPosts" and event.body.details is None for event in events)
            with closing(sqlite3.connect(run / "events.db")) as writer, writer:
                writer.execute("DELETE FROM events WHERE agent_id=3")
            assert record.feed_counts(1) is None


def test_offline_pathway_matches_persistent_checkpoint_and_trace(tmp_path, persistent_binary, probe_binary):
    metadata = reproducible_metadata(create_scenario_metadata(
        "pathway-fixture", influence=0.1, rewiring=0.1, repost_rate=0.15, recsys_type="OpinionM9", max_sim_step=2),
        ("typed-exposure-fixture",), node_count=24, follow_count=4)
    metadata = replace(metadata, CheckpointSteps=(2,),
                       RNG=Rng(Algorithm="pcg64-dxsm-v1", Seed1="0x0012", Seed2="0x0034"))
    fresh = observe_persistent(persistent_binary, tmp_path / "fresh", metadata, [0, 2])
    checkpoint = Path(fresh.run_directory) / "checkpoint-000000002.msgpack.lz4"
    measured = run_probe_checkpoints([str(checkpoint)], measurements=["pathway"], binary_path=str(probe_binary))
    (state,) = measured["results"]
    assert state["step"] == 2 and state["pathway"] == to_wire(fresh.endpoints[-1].pathway)
    metadata = replace(metadata, MaxSimulationStep=3)
    change = PolicyChange(at=0, recommender="StructureM9", bridge=True)
    ordinary = observe_persistent(persistent_binary, tmp_path / "ordinary", metadata, [0, 1, 3], policy_change=change)
    path = tmp_path / "detailed" / metadata.UniqueName / "feeds.jsonl.gz"
    detailed = observe_persistent(persistent_binary, tmp_path / "detailed", metadata, [0, 1, 3],
                                  policy_change=change, exposure_trace=path)
    assert detailed.endpoints == ordinary.endpoints
    assert detailed.final_opinions == ordinary.final_opinions
    assert sum(point.exposure.bridge.shown for point in detailed.endpoints) > 0
    rows = list(iter_exposure_trace(tmp_path, replace(detailed.exposure_trace, path=path.relative_to(tmp_path).as_posix())))
    for previous, current in zip(detailed.endpoints, detailed.endpoints[1:]):
        window = [row for row in rows if previous.step < row.step <= current.step]
        counts = current.exposure
        assert len(window) == counts.updates
        assert sum(row.requested for row in window) == counts.requested
        assert len({(row.step, row.recipient) for row in window}) == len(window)
        for role in ("exploration", "policy", "bridge"):
            slots = [(row, slot) for row in window for slot in row.recommendations if slot.source == role]
            assert len(slots) == getattr(counts, role).shown
            assert all(slot.placement is not None for _, slot in slots)
            cross = [(row, slot) for row, slot in slots
                     if row.opinion * slot.post.Opinion < 0
                     and min(abs(row.opinion), abs(slot.post.Opinion)) > 0.45 / 4]
            assert len(cross) == getattr(counts, role).cross_camp
            confident = [(row, slot) for row, slot in cross if abs(row.opinion - slot.post.Opinion) <= 0.45]
            assert len(confident) == getattr(counts, role).cross_confident
            eligible = [slot for row, slot in confident
                        if slot.post.AgentID != row.recipient
                        and slot.post.AgentID not in {neighbor.placement.holder for neighbor in row.neighbors}
                        and any(neighbor.post is not None and abs(row.opinion - neighbor.post.Opinion) > 0.45
                                for neighbor in row.neighbors)]
            assert len(eligible) == getattr(counts, role).cross_rewire
        assert sum(sum(slot.post is not None for slot in row.neighbors) for row in window) == counts.neighbor_seen


def test_persistent_smp_reads_enriched_view_posts(tmp_path):
    source = Path(__file__).resolve().parents[3] / "social-media-models"
    binary = tmp_path / "smp"
    subprocess.run(["go", "build", "-o", str(binary), "./cmd/smp"], cwd=source, check=True)
    metadata = create_scenario_metadata("persistent-fixture", max_sim_step=3)
    metadata = replace(metadata, NodeCount=16, NodeFollowCount=3, ViewPostsDetails=True,
                       RecsysFactoryType="OpinionM9",
                       RNG=Rng(Algorithm="pcg64-dxsm-v1", Seed1="0x0012", Seed2="0x0034"))
    subprocess.run([str(binary), str(tmp_path), json.dumps(to_wire(metadata))], capture_output=True, check=True)
    with RawSimulationRecord(str(tmp_path), metadata) as record:
        views = list(record.iter_feed_events(1, 2))
        assert len(views) == 16
        assert any(slot["source"] == "exploration" for event in views for slot in event.body.details["recommendations"])
        for view in views:
            details = view.body.details
            raw = record.feed_counts(view.step)[view.agent_id]
            assert len(details["recommendations"]) == raw[1] + raw[3]
            assert details["requested"] == metadata.RecsysCount
            assert all(slot["placement"] is not None for slot in details["recommendations"])
            assert sum(slot["post"] is not None for slot in details["neighbors"]) == raw[0] + raw[2]
            assert {slot["placement"]["holder"] for slot in details["neighbors"]} == set(record.get_graph(0).successors(view.agent_id))


@pytest.mark.parametrize("persisted_metadata", [False, True])
def test_drift_probe_uses_native_source_without_loading_raw_history(tmp_path, monkeypatch, persisted_metadata):
    from experiments.drift_probe.run import run_scenario
    from experiments.drift_probe.scenarios import LandscapeScenario

    scenario = LandscapeScenario("fixture", "legacy", 10, "fixture", 0.1, 0.0)
    directory = tmp_path / "legacy"
    directory.mkdir()
    if persisted_metadata:
        (directory / "metadata.json").write_text(json.dumps(scenario.metadata()))
    calls = []

    def observe(source, steps, **options):
        calls.append((source, steps, options))
        states = []
        for step in steps:
            points = []
            for x in (-1, 0, 1):
                summary = {"mean": 0, "variance": 0, "samples": 1, "active": 0}
                points.append({
                    "x": x, "f_probe": summary, "f_neighbor": summary, "f_recommendation": summary,
                    "mean_concordant_neighbor": 0, "mean_concordant_recommendation": 0,
                })
            states.append({"step": step, "points": points})
        return {"results": states, "rng": {}, "source_max_step": 12}

    monkeypatch.setattr("smp_bindings.run_probe_run", observe)
    path, manifest = run_scenario(
        scenario, workspace=tmp_path, probe_binary=tmp_path / "smp-probe", output_dir=tmp_path / "analysis",
        grid_step=1, replicates=1,
    )
    source, steps, options = calls[0]
    assert source == directory and steps == list(range(11))
    assert options["measurements"] == ["force"]
    assert options["metadata"] == (None if persisted_metadata else scenario.metadata())
    assert manifest["max_step"] == 12
    with np.load(path) as data:
        assert data["f_probe_mean"].shape == (11, 3)
