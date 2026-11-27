"""Partial recomputation figures must retain missing groups and signed values."""

import math
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.pathway_index.bar_plot import plot_grouped_bars, summarize_bar, write_bar_caption
from experiments.paper.pathway_index.plot import k_filters_recsys, k_labels_recsys


def row(name, iw, value, *, policy="OpinionM9", history=0):
    return ScenarioStatistics(name=f"s_grad_{name}", pathway_integral=iw,
                              recsys_type=policy, post_retain_count=history, g_index_mean_active=value)


def test_finite_metric_counts_exclude_missing_observations():
    result = summarize_bar([row("a", 0.2, -0.2), row("b", 0.2, math.nan)], (),
                           lambda record: record.g_index_mean_active)
    assert result.count == 1 and result.mean == -0.2 and result.std == 0


def test_partial_bars_use_one_explicit_cutoff_and_preserve_negative_values(monkeypatch):
    # A row predicate must not reopen dotenv or infer a hidden cutoff.
    monkeypatch.setattr("experiments.paper.analysis.selection.descriptive_iw_cutoff",
                        lambda: pytest.fail("row predicate read configuration"))
    rows = [row("low", 0.59, -0.2), row("high", 0.6, 1.2),
            row("history", 0.6, 100, history=2), row("unknown", math.nan, 100)]
    fig, ax = plt.subplots()
    try:
        lines = plot_grouped_bars(ax, lambda record: record.g_index_mean_active,
                                 records=rows, cutoff=0.6,
                                 category_filters=k_filters_recsys, category_labels=k_labels_recsys)
        heights = [patch.get_height() for patch in ax.patches]
        assert heights == [-0.2, 1.2]
        assert ax.get_ylim()[0] < -0.2 and ax.get_ylim()[1] > 1.2
        assert [label.get_text() for label in ax.get_xticklabels()] == ["St.", "Op."]
        assert all("n=" not in text.get_text() for text in ax.texts)
        assert sum("n=0" in line for line in lines) == 2
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_counts_are_exported_in_caption_only(tmp_path):
    write_bar_caption(tmp_path / "figure", [row("a", 0.4, 0.2)], 0.6, ["Op. - low: n=1"])
    caption = (tmp_path / "figure_caption.txt").read_text(encoding="utf-8")
    assert "n=1" in caption and "0.6" in caption and "population standard deviations" in caption


def test_empty_groups_render_without_zero_observations():
    fig, ax = plt.subplots()
    try:
        plot_grouped_bars(ax, lambda record: record.g_index_mean_active,
                          records=[], cutoff=0.6,
                          category_filters=k_filters_recsys, category_labels=k_labels_recsys)
        assert not ax.patches
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("module,expected", [
    ("plot", [(1e3, 1e4), (0., .4), (.3, .8), (1e3, 2e5)] * 2),
    ("plot_raw", [(1e3, 1e4), (0., .4), (.3, .8), (1e3, 2e5), (1e3, 2e5)]),
])
def test_bar_entry_points_restore_paper_limits(module, expected, tmp_path, monkeypatch):
    import numpy as np
    import experiments.paper.analysis.selection as selection
    import common.plotting as plotting

    records = [
        ScenarioStatistics(name=f"s_grad_{policy}_{repost}_{history}_{iw}", recsys_type=policy,
                           repost_rate=repost, post_retain_count=history, pathway_integral=iw,
                           event_count=4000, active_step=100, event_step_mean=20, triads=20000,
                           x_indices=np.array([0, 100]), g_index=np.array([0., 1.]))
        for policy in ("StructureM9", "OpinionM9") for repost in (0., .5)
        for history in (0, 2, 6) for iw in (.2, .8)
    ]
    monkeypatch.setattr(selection, "read_statistics", lambda *args, **kwargs: records)
    monkeypatch.setattr(selection, "descriptive_iw_cutoff", lambda: .6)
    monkeypatch.setenv("EHK_PATHWAY_OUTPUT_DIR", str(tmp_path))
    saved = []

    def save(figure, path):
        try:
            axes = [axis for axis in figure.axes if axis.get_visible()]
            saved.append([axis.get_ylim() for axis in axes])
            assert all("n=" not in text.get_text() for axis in axes for text in axis.texts)
            figure.canvas.draw()
        finally:
            plt.close(figure)

    monkeypatch.setattr(plotting, "plt_save_and_close", save)
    script = Path(__file__).resolve().parents[2] / "experiments/paper/pathway_index" / f"{module}.py"
    runpy.run_path(str(script), run_name="__main__")
    assert len(saved) == 1
    np.testing.assert_allclose(saved[0], expected)
