"""Recompute named contrasts for explicit history scopes and denominators."""

import numpy as np
import json
from dataclasses import replace
import pytest

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.pathway_index.grouped_summary import summarize


def test_combined_source_receipt_distinguishes_available_and_selected_counts(tmp_path):
    from experiments.paper.analysis.models import AnalysisProtocol
    from experiments.paper.analysis.records import StatisticsStore
    from ehk.metrics.protocol import METRIC_REVISION, metric_protocol
    from experiments.paper.pathway_index.grouped_summary import analyze

    database = tmp_path / "stats.sqlite"
    with StatisticsStore(database) as store:
        for origin, history in (("raw-a", 2), ("raw-b", 0)):
            store.put(ScenarioStatistics(
                name="s_grad_same_name", origin=origin, recsys_type="OpinionM9",
                post_retain_count=history, pathway_integral=.8, repost_rate=0,
                triads=100, event_count=10, complete=True, metric_revision=METRIC_REVISION,
                endpoint_complement=True), mode="gradation", analysis_id="fixture",
                protocol=AnalysisProtocol(metrics=metric_protocol()))
    result = analyze(database, tmp_path / "groups", cutoff=.6, origins=("raw-a", "raw-b"),
                     analysis_id="fixture", history_scope="zero")
    assert result["runs"] == 1
    receipt = json.loads((tmp_path / "groups/grouped_summary_receipt.json").read_text())
    assert receipt["schema"] == 2
    assert receipt["origins"] == {"raw-a": 1, "raw-b": 1}
    assert receipt["selected_origins"] == {"raw-b": 1}


def test_zero_scope_percentages_use_reference_means():
    records = []
    for policy, base in (("OpinionM9", 10), ("StructureM9", 20)):
        for group, addition, integral in (("low", 0, 0.4), ("high", 10, 0.8)):
            for repost in (0, 0.1):
                value = base + addition + (5 if repost else 0)
                records.append(
                    ScenarioStatistics(
                        name=f"s_grad_{policy}_{group}_{repost}",
                        origin="raw",
                        analysis_id="r3",
                        recsys_type=policy,
                        post_retain_count=0,
                        repost_rate=repost,
                        pathway_integral=integral,
                        event_count=value,
                        event_step_mean=5,
                        active_step=10,
                        x_indices=np.array([0, 5, 10]),
                        g_index=np.array([0.0, 0.5, 1.0]),
                        triads=100 * value,
                    )
                )
    history = replace(records[0])
    history.name, history.post_retain_count, history.event_count = "s_grad_history2", 2, 99999
    data, summary, contrasts = summarize([*records, history], 0.6)
    assert len(data) == 8 and set(data["origin"]) == {"raw"}
    assert set(summary["n_runs"]) == {2}
    assert set(summary.loc[summary.metric == "subjective_index_auc", "mean"]) == {0.5}
    contrast = contrasts.loc[
        (contrasts.comparison == "high_minus_low")
        & (contrasts.category == "policy:OpinionM9")
        & (contrasts.metric == "rewiring_events")
    ].iloc[0]
    assert contrast.target_mean == 22.5 and contrast.reference_mean == 12.5
    assert contrast.difference == 10 and contrast.relative_percent == 80
    repost = contrasts.loc[
        (contrasts.comparison == "reposting_on_minus_off")
        & (contrasts.group == "high")
        & (contrasts.metric == "rewiring_events")
    ].iloc[0]
    assert repost.relative_percent == 20
    with pytest.raises(ValueError, match="finite"):
        summarize(records, float("nan"))
    with pytest.raises(ValueError, match="no finite zero-history gradation records for the matched comparison"):
        summarize([], 0.6)


def test_original_design_pools_only_defined_opinion_histories_using_same_metrics():
    records = []
    for policy, history, low, high in (("StructureM9", 0, 10, 20),
                                       ("OpinionM9", 0, 10, 30),
                                       ("OpinionM9", 2, 20, 100),
                                       ("OpinionM9", 6, 30, 200),
                                       ("StructureM9", 2, 999, 999),
                                       ("OpinionM9", 4, 999, 999)):
        for group, integral, triads in (("low", .4, low), ("high", .8, high)):
            records.append(ScenarioStatistics(
                name=f"s_grad_{policy}_{history}_{group}", origin="raw", analysis_id="r6",
                recsys_type=policy, post_retain_count=history, repost_rate=0,
                pathway_integral=integral, event_count=10, event_step_mean=5,
                active_step=10, x_indices=np.array([0, 5, 10]),
                g_index=np.array([0., .5, 1.]), triads=triads,
            ))
    zero, _, _ = summarize(records, .6)
    original, summary, contrasts = summarize(records, .6, history_scope="original-design")
    assert len(zero) == 4 and len(original) == 8
    assert not original["name"].str.contains("StructureM9_2|OpinionM9_4").any()
    shared = zero.merge(original, on=["name", "origin"], suffixes=("_zero", "_original"))
    for metric in ("rewiring_events", "normalized_rewiring_time", "subjective_index_auc", "closed_triads"):
        np.testing.assert_array_equal(shared[metric+"_zero"], shared[metric+"_original"])
    high = summary[(summary.dimension == "policy") & (summary.category == "OpinionM9") &
                   (summary.group == "high") & (summary.metric == "closed_triads")].iloc[0]
    assert high.n_runs == 3 and high["mean"] == 110
    comparison = contrasts[(contrasts.comparison == "opinion_minus_structure") &
                           (contrasts.group == "high") & (contrasts.metric == "closed_triads")].iloc[0]
    assert comparison.target_mean == 110 and comparison.reference_mean == 20
    assert comparison.relative_percent == 450
    with pytest.raises(ValueError, match="history scope"):
        summarize([], .6, history_scope="undefined")
