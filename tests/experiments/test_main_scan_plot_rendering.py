"""Regression cases for partial-scan figures and their typed metric contract."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.main_scan import plot


def _row(*, value=1.3, strategy="OpinionM9", repost=0.0, peaks=2, history=0, origin="raw-fixture", active=9):
    return ScenarioStatistics(
        name="s_grad_partial", rewiring=0.005, decay=0.005,
        recsys_type=strategy, repost_rate=repost, post_retain_count=history,
        pathway_integral=value, active_step=active, last_opinion_peak_count=peaks,
        p_index=np.array([-0.2, 0.4]), p_backdrop=0.6,
        analysis_id="revision-fixture", origin=origin,
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
    plot.main([])
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
    assert "origins: raw-fixture" in combined_caption
    assert "source raw-fixture: snapshot n=1; selected history scope n=1" in combined_caption
    assert "history_scope: original-design" in combined_caption
    assert "snapshot: n=1" in combined_caption
    assert "$I_w$ (OpinionM9): n=1" in combined_caption
    assert "individual values below/above this scale: 0/1" in combined_caption
    assert "TV($I_p$), #P=1 (OpinionM9): n=0" in combined_caption
    assert "blank, not zero" in combined_caption
    assert "underlying values remain unchanged" in combined_caption
    active_caption = (tmp_path / "figures" / "f_p_active_step_caption.txt").read_text(encoding="utf-8")
    assert "(OpinionM9, p=0): n=1" in active_caption
    assert "(OpinionM9, p=0.5): n=0" in active_caption


@pytest.mark.parametrize(("scope", "expected"), [("original-design", .425), ("zero", .2)])
def test_history_scope_means_use_the_declared_design_only(scope, expected):
    rows = [_row(value=.2), _row(value=.2), _row(value=.5, history=2), _row(value=.8, history=6),
            _row(value=100, history=3), _row(value=100, strategy="Random"),
            _row(value=.1, strategy="StructureM9"), _row(value=100, strategy="StructureM9", history=2)]
    measure = lambda row: row.pathway_integral
    mean, _ = plot.create_heatmap_evaluator(rows, "OpinionM9", history_scope=scope)(measure)
    mean_retweet, _ = plot.create_heatmap_evaluator_retweet(rows, 0., "OpinionM9", history_scope=scope)(measure)
    mean_st, _ = plot.create_heatmap_evaluator(rows, "StructureM9", history_scope=scope)(measure)
    assert mean[0, 0] == pytest.approx(expected)
    assert mean_retweet[0, 0] == pytest.approx(expected)
    assert mean_st[0, 0] == pytest.approx(.1)
    default_mean, _ = plot.create_heatmap_evaluator(rows, "OpinionM9")(measure)
    assert default_mean[0, 0] == pytest.approx(.425)


@pytest.mark.parametrize(("scope", "expected_n", "expected_iw", "expected_active", "directory"), [
    ("original-design", 3, .5, np.mean(np.log10([9, 99, 999])), ""),
    ("zero", 1, .2, np.log10(9), "zero"),
])
def test_cli_applies_one_scope_to_heatmaps_summary_and_captions(
    tmp_path, monkeypatch, scope, expected_n, expected_iw, expected_active, directory,
):
    rows = [_row(value=.2, peaks=1), _row(value=.5, peaks=1, history=2, origin="other-source", active=99),
            _row(value=.8, peaks=1, history=6, origin="other-source", active=999)]
    reads = []

    def read(*args, **kwargs):
        reads.append((args, kwargs))
        return rows

    monkeypatch.setattr(plot, "read_statistics", read)
    saved = {}

    def save(figure, path):
        panels = [axis for axis in figure.axes if axis.images]
        saved[path] = [axis.images[0].get_array().copy() for axis in panels]
        plt.close(figure)

    monkeypatch.setattr(plot, "plt_save_and_close", save)
    database = tmp_path / "stats-v3.sqlite"
    arguments = ["--database", str(database), "--origin", "raw-fixture", "--origin", "other-source",
                 "--analysis-id", "revision-fixture", "--output", str(tmp_path)]
    if scope == "zero":
        arguments.extend(["--history-scope", scope])
    plot.main(arguments)
    assert reads == [((database,), {"mode": "gradation", "origins": ("raw-fixture", "other-source"),
                                   "analysis_id": "revision-fixture"})]
    figure_dir = tmp_path / directory / "figures"
    assert saved[str(figure_dir / "f_grad_and_pol_trad_index")][1][0, 0] == pytest.approx(expected_iw)
    assert saved[str(figure_dir / "f_p_active_step")][1][0, 0] == pytest.approx(expected_active)
    summary = (figure_dir / "f_p_backdrop_stats.txt").read_text(encoding="utf-8")
    assert f"p_index_op: n={expected_n}," in summary
    for stem in ("f_grad_and_pol_trad_index", "f_p_active_step"):
        caption = (figure_dir / f"{stem}_caption.txt").read_text(encoding="utf-8")
        assert "origins: other-source, raw-fixture" in caption
        assert "source raw-fixture: snapshot n=1; selected history scope n=1" in caption
        assert f"source other-source: snapshot n=2; selected history scope n={expected_n - 1}" in caption
        assert f"history_scope: {scope}" in caption
        assert "equal weight" in caption and "not paired effects" in caption
        assert ("History composition: OpinionM9, k=2: n=1" in caption) == (scope == "original-design")
    if scope == "zero":
        assert not (tmp_path / "figures").exists()


def test_empty_metric_list_is_rejected_before_allocating_a_figure():
    with pytest.raises(ValueError, match="at least one"):
        plot.plot_combined_heatmaps([], [])
