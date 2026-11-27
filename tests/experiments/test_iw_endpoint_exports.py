"""Keep stored pathway conventions visible in audits and grouped exports."""

import json

import pandas as pd
import pytest

from experiments.paper.analysis.records import ScenarioStatistics, StatisticsStore
from experiments.paper.analysis.selection import read_statistics, recorded_endpoint_complement
from experiments.paper.pathway_index.grouped_summary import summarize


def test_stored_conventions_are_not_replaced_by_the_new_default():
    assert recorded_endpoint_complement(ScenarioStatistics()) is None
    assert recorded_endpoint_complement(ScenarioStatistics(endpoint_complement=False)) is False
    assert recorded_endpoint_complement(
        ScenarioStatistics(protocol={"metrics": {"endpoint_complement": True}})
    ) is True
    with pytest.raises(ValueError, match="conflicting"):
        recorded_endpoint_complement(
            ScenarioStatistics(endpoint_complement=True, protocol={"metrics": {"endpoint_complement": False}})
        )


def test_selection_rejects_a_mixed_convention_even_under_one_analysis_id(tmp_path):
    database = tmp_path / "mixed.sqlite"
    with StatisticsStore(database) as store:
        for name, mode in (("a", True), ("b", False)):
            store.put(
                ScenarioStatistics(name=name, origin="raw", endpoint_complement=mode),
                analysis_id="same-id", mode="gradation",
            )
    with pytest.raises(ValueError, match="mixed endpoint-complement"):
        read_statistics(database, mode="gradation")


@pytest.mark.parametrize("endpoint_complement", [True, False])
def test_grouped_export_discloses_the_selected_convention(tmp_path, endpoint_complement):
    value = .9 if endpoint_complement else .1
    record = ScenarioStatistics(name="s_grad_fixture", origin="raw", recsys_type="OpinionM9",
        post_retain_count=0, decay=.05, rewiring=.05, repost_rate=.1, step=20,
        endpoint_complement=endpoint_complement, pathway_integral=value)
    database = tmp_path / "stats.sqlite"
    with StatisticsStore(database) as store:
        store.put(record, analysis_id="fixture", mode="gradation")
    rows = read_statistics(database, mode="gradation")
    grouped, _, _ = summarize(rows, .6)
    assert grouped["endpoint_complement"].item() == endpoint_complement
    assert grouped["I_w"].item() == value
