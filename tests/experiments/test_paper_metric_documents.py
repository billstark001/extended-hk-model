"""Numerical conventions, raw reconstruction, and typed document persistence."""

import sqlite3

import networkx as nx
import numpy as np
import pytest

from common.storage.documents import DocumentStore
from ehk.metrics.homophily import mean_followee_edge_concordance
from ehk.micro.observables import calc_active_step, get_gradation_index_hp, get_metric_audit
from experiments.paper.analysis.records import ScenarioStatistics, StatisticsStore
from experiments.paper.analysis.statistics import analysis_protocol
from experiments.paper.pathway_index.audit import analyze


def test_documents_roundtrip_arrays_metadata_nonfinite_and_atomic_upserts(tmp_path):
    path = tmp_path / "documents.sqlite"
    with DocumentStore(path) as store:
        value = {
            "metadata": {"RNG": {"Seed1": "18446744073709551615"}},
            "integer": np.array([1, 2], dtype=np.int32),
            "missing": float("nan"),
            "matrix": np.array([[0.2, np.inf]], dtype=np.float64),
        }
        store.put("analysis", "task", value)
        value["extra_statistic"] = 0.8
        store.put("analysis", "task", value)
        assert len(list(store.values("analysis"))) == 1
    with DocumentStore(path, readonly=True) as store:
        result = store.get("analysis", "task")
        assert result["integer"].dtype == np.int32
        assert np.isnan(result["missing"]) and np.isinf(result["matrix"][0, 1])
        assert result["metadata"] == value["metadata"] and result["extra_statistic"] == 0.8
        with pytest.raises(RuntimeError, match="read-only"):
            store.put("analysis", "task", {})
    with DocumentStore(path) as store, pytest.raises(ValueError):
        store.put("analysis", "unsafe", {"object": np.array([object()])})


def test_edge_threshold_and_signed_path_integral_follow_paper_definition():
    graph = nx.DiGraph([(0, 1)])
    assert mean_followee_edge_concordance(graph, np.array([0.0, 0.45]), 0.45) == 0
    assert get_gradation_index_hp(np.array([0, 0.5, 0.1]), np.array([0, 0, 1])) == pytest.approx(0.7)
    assert get_gradation_index_hp(
        np.array([0, 0.5, 0.1]), np.array([0, 0, 1]), endpoint_complement=False
    ) == pytest.approx(-0.2)
    assert get_gradation_index_hp(np.array([0, 0.2]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_activity_uses_terminal_reference_and_falls_back_if_never_crossed():
    steps = np.array([0, 10, 20])
    result = calc_active_step(steps, np.array([0.0, 1.0, 0.8]), 0.98, 0.75)
    assert result[1] == 7  # max(.98*.8,.75)=.784; historical max-over-time would give 9.
    absent = calc_active_step(steps, np.array([0.1, 0.2, 0.3]), 0.98, 0.75)
    assert absent[1] == 20 and absent[4] == pytest.approx(0.2)


def test_activity_already_above_criterion_starts_at_initial_observation():
    result = calc_active_step(np.array([0, 10, 20]), np.array([1.0, 1.0, 1.0]), 0.98, 0.75)
    assert result[:3] == (0, 0, 1.0)
    np.testing.assert_array_equal(result[3], [1.0])
    assert result[4] == 1.0


@pytest.mark.parametrize("endpoint_complement", [True, False])
def test_raw_metric_audit_preserves_missing_feed_and_separates_conventions(monkeypatch, tmp_path, endpoint_complement):
    from ehk.micro import observables

    class Raw:
        metadata = {"HKParams": {"Tolerance": 0.45}}

    monkeypatch.setattr(observables, "calc_distance_components", lambda raw, t: (t / 100, 0.2, 0.1 + t / 100, 0.3))
    steps = np.array([0, 10, 20])
    audit = get_metric_audit(
        Raw(),
        steps,
        np.array([0.0, 0.1, 0.2]),
        np.array([0.1, 0.2, 0.3]),
        np.array([0.0, 0.5, 1.0]),
        np.array([np.nan, 0.6, 0.8]),
        endpoint_complement=endpoint_complement,
    )
    assert audit["audit_prefix_start"] == 10
    assert np.array_equal(audit["audit_missing_steps"], [0])
    assert np.isnan(audit["legacy_h_index"][0])
    expected = 0.9 if endpoint_complement else 0.1
    assert audit["audit_endpoint_complement"] is endpoint_complement
    assert audit["pathway_variants"]["revised_full_trajectory"] == pytest.approx(expected)
    assert audit["pathway_variants"]["full_trajectory_with_complement"] == pytest.approx(0.9)
    assert audit["pathway_variants"]["full_trajectory_without_complement"] == pytest.approx(0.1)
    path = tmp_path / "stats.sqlite"
    with StatisticsStore(path) as store:
        store.put(
            ScenarioStatistics(
                name="case",
                origin="raw",
                recsys_type="OpinionM9",
                post_retain_count=0,
                decay=0.05,
                rewiring=0.05,
                repost_rate=0.1,
                step=20,
                pathway_integral=expected,
                endpoint_complement=endpoint_complement,
                **audit,
            ),
            analysis_id="fixture",
            mode="gradation",
        )
    report = analyze(path, [], tmp_path / "audit")
    assert report["partial_legacy_reconstructions"] == 1
    assert not report["full_campaign_validated"]
    assert "not a validated" in report["cutoff_status"]
    assert (tmp_path / "audit/pathway_group_migration.csv").is_file()


def test_document_statistics_reject_legacy_wide_tables(tmp_path):
    path = tmp_path / "archived.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE scenariostatistics(name TEXT)")
    with pytest.raises(ValueError, match="separate new"):
        StatisticsStore(path)
    with sqlite3.connect(path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"scenariostatistics"}


def test_recorded_exact_revision_remains_readable_after_binned_adoption(tmp_path):
    from ehk.metrics.protocol import METRIC_PROTOCOL

    path = tmp_path / "historical.sqlite"
    value = ScenarioStatistics(name="old", origin="raw", metric_revision=3, pathway_integral=.21844215537211242)
    with StatisticsStore(path) as store:
        store.put(value, analysis_id="original-exact", mode="gradation", protocol={"metrics": {"revision": 3}})
    with StatisticsStore(path, readonly=True) as store:
        old = store.get("old", "raw", "original-exact")
        assert old.metric_revision == 3 and old.protocol["metrics"]["revision"] == 3
        assert old.pathway_integral == .21844215537211242
        assert METRIC_PROTOCOL["distance_kde"]["bins"] == 4081


def test_optional_probe_binary_identity_is_required_only_when_enabled(tmp_path, monkeypatch):
    binary = tmp_path / "probe"
    monkeypatch.setenv("SMP_PROBE_BINARY", str(binary))
    assert "native_probe" not in analysis_protocol()
    binary.write_bytes(b"version one")
    first = analysis_protocol({"first_passage": {"cadence": 16}})
    binary.write_bytes(b"version two")
    second = analysis_protocol({"first_passage": {"cadence": 16}})
    assert first["native_probe"]["sha256"] != second["native_probe"]["sha256"]
