"""Figure readers require explicit compatible sources and declared design scopes."""

from dataclasses import replace

import numpy as np
import pytest
from experiments.paper.analysis.models import AnalysisProtocol
from ehk.metrics.protocol import METRIC_REVISION, metric_protocol

from experiments.paper.analysis.records import ScenarioStatistics, StatisticsStore
from experiments.paper.analysis.selection import is_high_iw, is_low_iw, read_statistics


@pytest.fixture(autouse=True)
def clear_source_selection(monkeypatch):
    for key in ("EHK_STATS_ORIGIN", "EHK_STATS_ORIGINS", "EHK_ANALYSIS_ID"):
        monkeypatch.delenv(key, raising=False)


def _record(name="s_grad_fixture", *, origin="raw-a", history=0, value=0.4):
    return ScenarioStatistics(
        name=name,
        origin=origin,
        tolerance=0.45,
        decay=0.05,
        rewiring=0.05,
        repost_rate=0.0,
        recsys_type="OpinionM9",
        post_retain_count=history,
        pathway_integral=value,
        x_indices=np.array([0, 8]),
        p_index=np.array([0.0, 0.8]),
        h_index=np.array([0.0, 0.5]),
        complete=True,
        metric_revision=METRIC_REVISION,
        endpoint_complement=True,
    )


def test_figure_selection_requires_revision_and_origin_when_ambiguous(tmp_path, monkeypatch):
    monkeypatch.delenv("EHK_ANALYSIS_ID", raising=False)
    monkeypatch.delenv("EHK_STATS_ORIGIN", raising=False)
    database = tmp_path / "stats.sqlite"
    with StatisticsStore(database) as store:
        for origin, revision in (("raw-a", "r1"), ("raw-a", "r2"), ("raw-b", "r2")):
            store.put(_record(origin=origin), analysis_id=revision, mode="gradation", protocol=AnalysisProtocol(metrics=metric_protocol()))
        store.put(_record("s_eps_fixture"), analysis_id="epsilon-r1", mode="epsilon", protocol=AnalysisProtocol(metrics=metric_protocol()))
    with pytest.raises(ValueError, match="multiple analysis revisions"):
        read_statistics(database, mode="gradation")
    with pytest.raises(ValueError, match="multiple source datasets"):
        read_statistics(database, mode="gradation", analysis_id="r2")
    rows = read_statistics(database, mode="gradation", analysis_id="r2", origin="raw-a")
    assert len(rows) == 1 and rows[0].analysis_id == "r2"
    np.testing.assert_array_equal(rows[0].p_index, [0.0, 0.8])


def test_disjoint_partial_sources_are_not_silently_pooled(tmp_path):
    database = tmp_path / "partial.sqlite"
    with StatisticsStore(database) as store:
        store.put(_record("s_grad_a", origin="raw-a"), analysis_id="r1", mode="gradation", protocol=AnalysisProtocol(metrics=metric_protocol()))
        store.put(_record("s_grad_b", origin="raw-b"), analysis_id="r1", mode="gradation", protocol=AnalysisProtocol(metrics=metric_protocol()))
    with pytest.raises(ValueError, match="multiple source datasets"):
        read_statistics(database, mode="gradation", analysis_id="r1")


def test_explicit_multiple_sources_preserve_same_name_runs_and_environment(tmp_path, monkeypatch):
    database = tmp_path / "combined.sqlite"
    with StatisticsStore(database) as store:
        for origin in ("raw-a", "raw-b", "unselected"):
            store.put(_record(origin=origin), analysis_id="r1", mode="gradation",
                      protocol=AnalysisProtocol(metrics=metric_protocol()))
    rows = read_statistics(database, mode="gradation", origins=("raw-a", "raw-b"), analysis_id="r1")
    assert sorted((row.origin, row.name) for row in rows) == [
        ("raw-a", "s_grad_fixture"), ("raw-b", "s_grad_fixture")]
    monkeypatch.setenv("EHK_STATS_ORIGINS", "raw-a, raw-b")
    assert len(read_statistics(database, mode="gradation", analysis_id="r1")) == 2
    # An explicit one-source request overrides the environment.
    assert len(read_statistics(database, mode="gradation", origin="raw-a", analysis_id="r1")) == 1
    monkeypatch.setenv("EHK_STATS_ORIGIN", "raw-a")
    with pytest.raises(ValueError, match="not both"):
        read_statistics(database, mode="gradation", analysis_id="r1")


@pytest.mark.parametrize("origins, message", [
    ((), "nonempty"), (("raw-a", ""), "nonempty"),
    (("raw-a", "raw-a"), "duplicate"), (("raw-a", "missing"), "no matching"),
    ("raw-a", "not a string"),
])
def test_multiple_sources_reject_incomplete_or_ambiguous_requests(tmp_path, origins, message):
    database = tmp_path / "selection.sqlite"
    with StatisticsStore(database) as store:
        store.put(_record(), analysis_id="r1", mode="gradation", protocol=AnalysisProtocol(metrics=metric_protocol()))
    with pytest.raises(ValueError, match=message):
        read_statistics(database, mode="gradation", origins=origins)


def test_multiple_sources_reject_same_revision_with_incompatible_protocols(tmp_path):
    database = tmp_path / "protocol.sqlite"
    protocol = AnalysisProtocol(metrics=metric_protocol())
    with StatisticsStore(database) as store:
        store.put(_record(origin="raw-a"), analysis_id="r1", mode="gradation", protocol=protocol)
        store.put(_record(origin="raw-b"), analysis_id="r1", mode="gradation",
                  protocol=replace(protocol, active_threshold=.5))
    with pytest.raises(ValueError, match="incompatible recorded analysis protocols"):
        read_statistics(database, mode="gradation", origins=("raw-a", "raw-b"))


def test_invalidated_source_statistics_cannot_be_used_in_a_figure(tmp_path):
    database = tmp_path / "stale.sqlite"
    row = _record()
    row.complete, row.stale = False, True
    with StatisticsStore(database) as store:
        store.put(row, analysis_id="r1", mode="gradation", protocol=AnalysisProtocol(metrics=metric_protocol()))
    with pytest.raises(ValueError, match="stale or incomplete"):
        read_statistics(database, mode="gradation")


def test_main_and_raw_pool_original_design_but_keep_explicit_history_controls():
    from experiments.paper.main_scan.plot import create_heatmap_evaluator
    from experiments.paper.pathway_index.bar_plot import summarize_bar
    from experiments.paper.pathway_index.plot_raw import k_filters_1, k_filters_2

    rows = [
        _record(f"s_grad_fixture_{history}", history=history, value=value)
        for history, value in ((0, 0.2), (2, 0.7), (6, 1.2))
    ]
    mean, _ = create_heatmap_evaluator(rows, "OpinionM9")(lambda row: row.pathway_integral)
    assert np.count_nonzero(np.isfinite(mean)) == 1
    assert mean[np.isfinite(mean)].item() == pytest.approx(0.7)
    mean_zero, _ = create_heatmap_evaluator(rows, "OpinionM9", history_scope="zero")(lambda row: row.pathway_integral)
    assert mean_zero[np.isfinite(mean_zero)].item() == 0.2
    summary = summarize_bar(rows, k_filters_1[2], lambda row: row.pathway_integral)
    assert summary.mean == pytest.approx(.7) and summary.count == 3
    for filters, expected in ((k_filters_2[1], 0.7), (k_filters_2[2], 1.2)):
        summary = summarize_bar(rows, filters, lambda row: row.pathway_integral)
        assert summary.mean == expected and summary.std == 0 and summary.count == 1


def test_grouped_figures_require_explicit_descriptive_cutoff(monkeypatch):
    monkeypatch.delenv("EHK_IW_CUTOFF", raising=False)
    with pytest.raises(ValueError, match="set EHK_IW_CUTOFF"):
        is_low_iw(_record())
    monkeypatch.setenv("EHK_IW_CUTOFF", "0.6")
    assert is_low_iw(_record(value=-0.1))
    assert is_high_iw(_record(value=0.6))
    assert is_high_iw(_record(value=1.4))
    assert not is_low_iw(_record(value=float("nan")))
    assert not is_high_iw(_record(value=None))
