"""Partial recomputation figures must retain missing groups and signed values."""

import math
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.pathway_index.bar_plot import (
    figure5_axis_limits, plot_grouped_bars, render_figure5, summarize_bar, write_bar_caption,
)
from experiments.paper.pathway_index.grouped_summary import METRICS, summarize
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
                                 category_filters=[(*filters, lambda record: record.post_retain_count == 0)
                                                   for filters in k_filters_recsys],
                                 category_labels=k_labels_recsys)
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
    rows = [row("a", 0.4, 0.2), row("b", 0.4, 0.2)]
    rows[0].origin, rows[1].origin = "source-a", "source-b"
    write_bar_caption(tmp_path / "figure", rows, 0.6, ["Op. - low: n=2"])
    caption = (tmp_path / "figure_caption.txt").read_text(encoding="utf-8")
    assert "n=2" in caption and "0.6" in caption and "population standard deviations" in caption
    assert "source-a: n=1" in caption and "source-b: n=1" in caption
    assert "match the existing paper" not in caption


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


def test_raw_bar_entry_point_keeps_outside_historical_ranges_visible(tmp_path, monkeypatch):
    import experiments.paper.analysis.selection as selection
    import common.plotting as plotting

    records = [
        ScenarioStatistics(name=f"s_grad_{policy}_{repost}_{history}_{iw}", recsys_type=policy,
                           repost_rate=repost, post_retain_count=history, pathway_integral=iw,
                           event_count=40000, active_step=100, event_step_mean=200, triads=400000,
                           x_indices=np.array([0, 100]), g_index=np.array([-.3, -.3]))
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
    script = Path(__file__).resolve().parents[2] / "experiments/paper/pathway_index/plot_raw.py"
    monkeypatch.setattr("sys.argv", [str(script)])
    runpy.run_path(str(script), run_name="__main__")
    assert len(saved) == 1
    for (lower, upper), value in zip(saved[0], (40000, 2., -.3, 400000, 400000)):
        assert lower < value < upper


def _figure5_rows():
    return [
        ScenarioStatistics(
            name=f"s_grad_{policy}_{repost}_{history}_{iw}", recsys_type=policy,
            origin="source", analysis_id="fixture", repost_rate=repost,
            post_retain_count=history, pathway_integral=iw,
            event_count=40000 + 3000 * history + 10000 * repost,
            active_step=100, event_step_mean=90 + 100 * repost,
            triads=300000 + 100000 * history + 10000 * repost,
            x_indices=np.array([0, 50, 100]), g_index=np.array([-.4, -.3, -.2]),
        )
        for policy in ("StructureM9", "OpinionM9") for repost in (0., .5)
        for history in (0, 2, 6) for iw in (.2, .8)
    ]


@pytest.mark.parametrize("scope,expected", [(None, 16), ("zero", 8)])
def test_formal_entry_point_selects_declared_design_and_keeps_all_error_endpoints(
    scope, expected, tmp_path, monkeypatch,
):
    from experiments.paper.pathway_index import plot

    reads, saved = [], []
    records = _figure5_rows()

    def read(*args, **kwargs):
        reads.append((args, kwargs))
        return records

    def save(figure, path):
        figure.canvas.draw()
        saved.append((path, [axis.get_ylim() for axis in figure.axes],
                      [[patch.get_height() for patch in axis.patches] for axis in figure.axes]))
        assert all("n=" not in text.get_text() for axis in figure.axes for text in axis.texts)
        plt.close(figure)

    monkeypatch.setattr(plot, "read_statistics", read)
    monkeypatch.setattr(plot, "plt_save_and_close", save)
    monkeypatch.setattr(plot, "descriptive_iw_cutoff", lambda: pytest.fail("explicit cutoff read environment"))
    args = ["--database", "fixture.sqlite", "--output", str(tmp_path), "--cutoff", ".6",
            "--analysis-id", "fixture", "--origin", "source-a", "--origin", "source-b"]
    if scope:
        args += ["--history-scope", scope]
    result = plot.main(args)
    assert result["runs"] == expected
    assert reads[0][1] == dict(mode="gradation", analysis_id="fixture", origins=("source-a", "source-b"))
    assert len(reads) == len(saved) == 1
    assert Path(saved[0][0]).name == "f_grad_index_interpret"
    output = (tmp_path / "zero" if scope == "zero" else tmp_path) / "figures"
    assert Path(saved[0][0]).parent == output
    summary = pd.read_csv(output / "f_grad_index_interpret_summary.csv")
    high = summary[(summary.category == "OpinionM9") & (summary.group == "high")]
    assert set(high.n_runs) == {6 if scope is None else 2}
    for col, metric in enumerate(METRICS):
        endpoints = summary[summary.metric == metric]
        lower = (endpoints["mean"] - endpoints.population_std).min()
        upper = (endpoints["mean"] + endpoints.population_std).max()
        for row_index in range(2):
            lo, hi = saved[0][1][row_index * 4 + col]
            assert lo < lower <= upper < hi
    assert all(height < 0 for height in saved[0][2][2])
    caption = (output / "f_grad_index_interpret_caption.txt").read_text(encoding="utf-8")
    assert f"History scope: {scope or 'original-design'}" in caption
    assert f"n={expected}" in caption and "weight one" in caption
    assert "ddof=0" in caption and "not confidence intervals" in caption
    assert "Selected configuration counts" in caption and "StructureM9, kh=2" not in caption
    assert ("OpinionM9, kh=2" in caption) == (scope is None)
    assert "No fixed historical y-limit" in caption
    assert f"Selected source counts: source: n={expected}" in caption


def test_shared_renderer_preserves_missing_groups_and_metric_missingness():
    record = _figure5_rows()[12]
    record.active_step = 0
    record.event_count = 0
    _, summary, _ = summarize([record], .6, history_scope="original-design")
    figure, axes = render_figure5(summary)
    try:
        assert len(axes[0, 0].patches) == 1
        assert axes[0, 0].patches[0].get_height() == 0
        assert not axes[0, 1].patches and not axes[0, 2].patches
        assert all(not axis.get_title().startswith("n=") for axis in figure.axes)
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_audit_figure5_uses_same_renderer_and_cross_cutoff_limits(tmp_path, monkeypatch):
    from experiments.paper.pathway_index import bar_plot
    from experiments.paper.pathway_metrics import figure5
    import common.plotting as plotting

    _, summary, _ = summarize(_figure5_rows(), .6, history_scope="original-design")
    extreme = summary.copy()
    extreme.loc[extreme.metric == "subjective_index_auc", "mean"] = 2.
    limits = figure5_axis_limits(pd.concat([summary, extreme], ignore_index=True))
    assert limits["subjective_index_auc"]["limits"][0] < -.3
    assert limits["subjective_index_auc"]["limits"][1] > 2.
    rendered = []
    actual_renderer = bar_plot.render_figure5

    def render(*args, **kwargs):
        rendered.append((args[0], kwargs["limits"]))
        return actual_renderer(*args, **kwargs)

    def save(figure, path):
        assert path == str(tmp_path / "figure5_iw_0.6")
        figure.canvas.draw()
        plt.close(figure)

    monkeypatch.setattr(bar_plot, "render_figure5", render)
    monkeypatch.setattr(plotting, "plt_save_and_close", save)
    figure5.plot(summary, .6, limits, tmp_path)
    assert len(rendered) == 1 and rendered[0][0] is summary and rendered[0][1] is limits
