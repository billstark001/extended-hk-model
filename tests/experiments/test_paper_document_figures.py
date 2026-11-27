"""Figure readers must not pool incompatible reanalyses or unmatched histories."""

import numpy as np
import pytest
from experiments.paper.analysis.models import AnalysisProtocol
from ehk.metrics.protocol import METRIC_REVISION, metric_protocol

from experiments.paper.analysis.records import ScenarioStatistics, StatisticsStore
from experiments.paper.analysis.selection import is_high_iw, is_low_iw, read_statistics


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


def test_invalidated_source_statistics_cannot_be_used_in_a_figure(tmp_path):
    database = tmp_path / "stale.sqlite"
    row = _record()
    row.complete, row.stale = False, True
    with StatisticsStore(database) as store:
        store.put(row, analysis_id="r1", mode="gradation", protocol=AnalysisProtocol(metrics=metric_protocol()))
    with pytest.raises(ValueError, match="stale or incomplete"):
        read_statistics(database, mode="gradation")


def test_main_heatmap_matches_history_but_raw_bars_keep_history_controls():
    from experiments.paper.main_scan.plot import create_heatmap_evaluator
    from experiments.paper.pathway_index.plot_raw import evaluate_bar_on_filters, k_filters_1, k_filters_2

    rows = [
        _record(f"s_grad_fixture_{history}", history=history, value=value)
        for history, value in ((0, 0.2), (2, 0.7), (6, 1.2))
    ]
    mean, _ = create_heatmap_evaluator(rows, "OpinionM9")(lambda row: row.pathway_integral)
    assert np.count_nonzero(np.isfinite(mean)) == 1
    assert mean[np.isfinite(mean)].item() == 0.2
    for filters, expected in ((k_filters_1[2], 0.2), (k_filters_2[1], 0.7), (k_filters_2[2], 1.2)):
        mean, std = evaluate_bar_on_filters(rows, list(filters), lambda row: row.pathway_integral)
        assert mean == expected and std == 0


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
