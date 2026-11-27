"""Recompute named contrasts with matched histories and explicit denominators."""

import numpy as np
from dataclasses import replace
import pytest

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.pathway_index.grouped_summary import summarize


def test_grouped_percentages_use_reference_means_and_exclude_history_confounding():
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
