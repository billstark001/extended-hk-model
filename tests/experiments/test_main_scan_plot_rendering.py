"""Regression cases for partial-scan figures and their typed metric contract."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.main_scan import plot


def _row(*, value=1.3, strategy="OpinionM9", repost=0.0, peaks=2):
    return ScenarioStatistics(
        name="s_grad_partial", rewiring=0.005, decay=0.005,
        recsys_type=strategy, repost_rate=repost, post_retain_count=0,
        pathway_integral=value, active_step=9, last_opinion_peak_count=peaks,
        p_index=np.array([-0.2, 0.4]), p_backdrop=0.6,
        analysis_id="revision-fixture", origin="raw-fixture",
    )


@pytest.mark.parametrize(("metric_count", "with_diff"), [(1, False), (1, True), (3, True)])
def test_layout_matches_metric_count_and_keeps_unobserved_cells_blank(metric_count, with_diff):
    rows = [_row()]
    metrics = [plot.HeatmapMetric(lambda row: row.pathway_integral, "$I_w$", plot.IW_SCALE, plot.IW_DIFFERENCE_SCALE)] * metric_count
    figure = plot.plot_combined_heatmaps(rows, metrics, with_diff=with_diff, with_std=False)
    try:
        panels = [axis for axis in figure.axes if axis.images]
        columns = 3 if with_diff else 2
        assert len(panels) == columns * metric_count
        assert panels[1].images[0].get_clim() == (0.0, 1.0)
        array = panels[1].images[0].get_array()
        assert np.ma.count(array) == 1
        assert float(array[0, 0]) == 1.3
        assert all(np.ma.is_masked(panels[0].images[0].get_array()[index]) for index in np.ndindex((8, 8)))
        assert not figure.texts
        assert all("n=" not in axis.get_title(loc="left") for axis in panels)
    finally:
        plt.close(figure)


def test_retweet_heatmap_keeps_paper_limits_without_modifying_outside_values():
    metric = plot.HeatmapMetric(lambda row: float(np.log10(row.active_step)), "$t$", plot.ACTIVE_STEP_SCALE)
    figure = plot.plot_heatmap_by_retweet([_row()], metric)
    try:
        panels = [axis for axis in figure.axes if axis.images]
        assert panels[1].images[0].get_clim() == (1.0, 5.0)
        assert float(panels[1].images[0].get_array()[0, 0]) == np.log10(9)
        assert panels[1].images[0].get_cmap().name == "YlGnBu"
        assert all(tuple(axis.get_yticks()) == (1.0, 2.0, 3.0, 4.0, 5.0)
                   for axis in figure.axes if not axis.images)
        assert [axis.get_title(loc="left")[:3] for axis in panels] == ["(a)", "(b)", "(c)", "(d)"]
        assert not figure.texts
    finally:
        plt.close(figure)


def test_main_uses_one_read_and_renders_when_consensus_subset_is_empty(tmp_path, monkeypatch, capsys):
    reads = []

    def read(*args, **kwargs):
        reads.append((args, kwargs))
        return [_row()]

    monkeypatch.setenv("EHK_MAIN_SCAN_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(plot, "read_statistics", read)
    saved = []
    rendered_scales = []

    def save(figure, path):
        saved.append(path)
        rendered_scales.append(([(axis.images[0].get_cmap().name, axis.images[0].get_clim())
                                 for axis in figure.axes if axis.images],
                                [tuple(axis.get_yticks()) for axis in figure.axes if not axis.images]))
        figure.canvas.draw()
        plt.close(figure)

    monkeypatch.setattr(plot, "plt_save_and_close", save)
    plot.main()
    assert len(reads) == 1
    assert len(saved) == 2
    scales, ticks = rendered_scales[0]
    assert scales == [("RdYlBu_r", (0.0, 1.0)), ("RdYlBu_r", (0.0, 1.0)), ("RdBu_r", (-0.6, 0.6)),
                      ("YlGnBu", (0.5, 2.0)), ("YlGnBu", (0.5, 2.0)), ("RdBu_r", (-1.0, 1.0))]
    assert ticks == [plot.IW_SCALE.ticks, plot.IW_SCALE.ticks, plot.IW_DIFFERENCE_SCALE.ticks,
                     plot.TV_SCALE.ticks, plot.TV_SCALE.ticks, plot.TV_DIFFERENCE_SCALE.ticks]
    summary = (tmp_path / "figures" / "f_p_backdrop_stats.txt").read_text(encoding="utf-8")
    assert summary.count("n=0; no matching finite observations") == 2
    assert summary.strip() == capsys.readouterr().out.strip()
    combined_caption = (tmp_path / "figures" / "f_grad_and_pol_trad_index_caption.txt").read_text(encoding="utf-8")
    assert "analysis_id: revision-fixture" in combined_caption
    assert "origin: raw-fixture" in combined_caption
    assert "snapshot: n=1" in combined_caption
    assert "$I_w$ (OpinionM9): n=1" in combined_caption
    assert "individual values below/above this scale: 0/1" in combined_caption
    assert "TV($I_p$), #P=1 (OpinionM9): n=0" in combined_caption
    assert "blank, not zero" in combined_caption
    assert "underlying values remain unchanged" in combined_caption
    active_caption = (tmp_path / "figures" / "f_p_active_step_caption.txt").read_text(encoding="utf-8")
    assert "(OpinionM9, p=0): n=1" in active_caption
    assert "(OpinionM9, p=0.5): n=0" in active_caption


def test_empty_metric_list_is_rejected_before_allocating_a_figure():
    with pytest.raises(ValueError, match="at least one"):
        plot.plot_combined_heatmaps([], [])
