"""The zero-history auxiliary bars must retain their separate history control."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.pathway_index import plot_raw


def test_zero_scope_limits_main_bars_but_keeps_opinion_history_control(tmp_path, monkeypatch):
    records = []
    for policy, history, count, timing, auc, triads in (
        ("StructureM9", 0, 20, .2, .2, 50),
        ("OpinionM9", 0, 10, .1, .1, 100),
        ("OpinionM9", 2, 200, .5, .5, 200),
        ("OpinionM9", 6, 600, .9, .9, 600),
        ("StructureM9", 2, 99999, 99., 99., 99999),
        ("OpinionM9", 4, 99999, 99., 99., 99999),
    ):
        records.append(ScenarioStatistics(
            name=f"s_grad_{policy}_{history}", origin="source", analysis_id="fixture",
            recsys_type=policy, post_retain_count=history, repost_rate=0., pathway_integral=.8,
            event_count=count, active_step=100, event_step_mean=100 * timing, triads=triads,
            x_indices=np.array([0., 100.]), g_index=np.array([auc, auc]),
        ))
    reads, saved = [], []

    def read(*args, **kwargs):
        reads.append((args, kwargs))
        return records

    def save(figure, path):
        try:
            panels = [axis for axis in figure.axes if axis.get_visible()]
            saved.append((Path(path), [[bar.get_height() for bar in axis.patches] for axis in panels]))
        finally:
            plt.close(figure)

    monkeypatch.setattr(plot_raw, "read_statistics", read)
    monkeypatch.setattr(plot_raw, "plt_save_and_close", save)
    monkeypatch.setattr(plot_raw, "descriptive_iw_cutoff", lambda: pytest.fail("explicit cutoff must take precedence"))
    plot_raw.main(["--database", "fixture.sqlite", "--origin", "source", "--analysis-id", "fixture",
                   "--output", str(tmp_path), "--cutoff", ".6", "--history-scope", "zero"])

    assert len(reads) == len(saved) == 1
    assert reads[0][0] == (Path("fixture.sqlite"),)
    assert reads[0][1] == {"mode": "gradation", "origins": ["source"], "analysis_id": "fixture"}
    output = tmp_path / "zero" / "figures"
    assert saved[0][0] == output / "f_grad_index_interpret_raw"
    assert not (tmp_path / "figures").exists()
    for actual, expected in zip(saved[0][1], ([20, 10], [.2, .1], [.2, .1], [50, 100], [100, 200, 600])):
        assert actual == pytest.approx(expected)
    assert len(saved[0][1]) == 5
    caption = (output / "f_grad_index_interpret_raw_caption.txt").read_text(encoding="utf-8")
    assert "Panels a-d history scope: zero; selected n=2" in caption
    assert "Panel e always shows separate OpinionM9 kh=0/2/6" in caption
    assert "k=2 - high" in caption and "k=6 - high" in caption
