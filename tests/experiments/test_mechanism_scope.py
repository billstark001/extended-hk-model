"""Formal reposting plots preserve declared histories, cutoffs and spread."""

from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.mechanism_cases import plot_cluster_and_repost as plot


@pytest.fixture
def records():
    rows = []
    for index, (policy, history) in enumerate((("StructureM9", 0), ("OpinionM9", 0),
                                              ("OpinionM9", 2), ("OpinionM9", 6))):
        for repost in (0., .1):
            for integral in (.2, .65, .8):
                rows.append(ScenarioStatistics(
                    name=f"s_grad_{policy}_{history}_{repost}_{integral}",
                    origin="raw-a" if index % 2 == 0 else "raw-b", analysis_id="fixture",
                    recsys_type=policy, post_retain_count=history, repost_rate=repost,
                    pathway_integral=integral, last_opinion_peak_count=1 if integral == .2 else 2,
                    opinion_diff_seg_mean=np.full(5, 1.2 if history == 6 else .05),
                    opinion_diff_seg_std=np.full(5, .5),
                ))
    rows.extend((replace(rows[0], name="s_grad_invalid_st_history", post_retain_count=2),
                 replace(rows[6], name="s_grad_invalid_op_history", post_retain_count=4)))
    return rows


def test_cli_defaults_to_original_design_and_captures_explicit_cutoff(records, tmp_path, monkeypatch):
    calls, figures = [], []
    monkeypatch.setenv("EHK_IW_CUTOFF", ".6")
    monkeypatch.setattr(plot, "read_statistics", lambda database, **kwargs: calls.append((database, kwargs)) or records)

    def save(figure, path):
        figures.append((path, figure.axes[3].get_ylim(), figure.axes[4].get_ylim()))
        plt.close(figure)

    monkeypatch.setattr(plot, "plt_save_and_close", save)
    result = plot.main(["--database", str(tmp_path / "source.sqlite"), "--origin", "raw-a", "--origin", "raw-b",
                        "--analysis-id", "fixture", "--output", str(tmp_path), "--cutoff", ".7"])
    assert result["history_scope"] == "original-design" and result["runs"] == 24
    assert calls[0][1]["origins"] == ("raw-a", "raw-b")
    caption = Path(result["output"] + "_caption.txt").read_text()
    assert "raw-a: n=12" in caption and "raw-b: n=12" in caption
    assert "OpinionM9/k=6: n=6" in caption and "StructureM9/k=0: n=6" in caption
    assert "low/p=0: n=8" in caption  # I_w=.65 follows --cutoff .7, not env .6.
    assert "not paired intervention effects" in caption
    assert all(lower < 0 and upper > .6 for lower, upper in figures[0][1:])


def test_zero_history_is_a_separate_output_and_environment_cutoff_is_used(records, tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "read_statistics", lambda *args, **kwargs: records)
    monkeypatch.setattr(plot, "plt_save_and_close", lambda fig, path: plt.close(fig))
    monkeypatch.setenv("EHK_IW_CUTOFF", ".6")
    result = plot.main(["--output", str(tmp_path), "--history-scope", "zero"])
    assert result["runs"] == 12 and result["cutoff"] == .6
    assert Path(result["output"]).parent == tmp_path / "zero" / "figures"
    caption = Path(result["output"] + "_caption.txt").read_text()
    assert "OpinionM9/k=2" not in caption and "OpinionM9/k=6" not in caption


def test_within_between_run_variance_definition_is_unchanged(records):
    first = replace(records[0], opinion_diff_seg_mean=np.zeros(5), opinion_diff_seg_std=np.ones(5))
    second = replace(records[0], opinion_diff_seg_mean=np.full(5, 2.), opinion_diff_seg_std=np.ones(5))
    mean, spread = plot.pooled_summary([first, second])
    np.testing.assert_allclose(mean, np.ones(5))
    np.testing.assert_allclose(spread, np.full(5, np.sqrt(2)))


def test_missing_selected_or_window_data_is_explicit(records):
    with pytest.raises(ValueError, match="no finite gradation"):
        plot.render_cluster_and_repost([], .6)
    with pytest.raises(ValueError, match="no five-window"):
        plot.render_cluster_and_repost([replace(row, opinion_diff_seg_mean=None) for row in records], .6)
    with pytest.raises(ValueError, match="invalid five-window"):
        plot.render_cluster_and_repost([replace(records[0], opinion_diff_seg_std=np.zeros(4))], .6)


def test_uncertainty_range_keeps_negative_lower_bound():
    figure, axis = plt.subplots()
    plot.plot_bar(axis, [[(.2, .5)]], ["window"], ["group"], ["tab:blue"])
    lower, upper = axis.get_ylim()
    plt.close(figure)
    assert lower < -.3 and upper > .7


@pytest.mark.parametrize("reposting", [False, True])
def test_moving_band_uses_mean_support_and_evaluation_grid(records, monkeypatch, reposting):
    real_smoother = plot.adaptive_moving_stats
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return real_smoother(*args, **kwargs)

    monkeypatch.setattr(plot, "adaptive_moving_stats", capture)
    rows = [replace(row, opinion_diff_seg_std=np.full(5, row.pathway_integral**2))
            for row in records]
    global_min, global_max = -.4, 1.4  # Deliberately wider than the reposting data support.
    grid, _, spread = plot._moving_curve(rows, reposting, global_min, global_max)
    subset = [row for row in rows if (row.repost_rate != 0) == reposting]
    x = np.array([row.pathway_integral for row in subset])
    std = np.array([row.opinion_diff_seg_std[1] for row in subset])
    support = {} if reposting else {"min": global_min, "max": global_max}
    assert calls[0] == support
    assert calls[1]["result_x"] is grid
    assert {key: value for key, value in calls[1].items() if key != "result_x"} == support
    _, expected, _ = real_smoother(x, std, .2, **support, result_x=grid)
    np.testing.assert_allclose(spread, expected)
    if reposting:
        assert grid[0] == x.min() and grid[-1] == x.max()
        _, mismatched, _ = real_smoother(x, std, .2, min=global_min, max=global_max)
        assert not np.allclose(spread, mismatched)
