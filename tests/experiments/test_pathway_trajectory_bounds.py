"""Paper viewports leave signed scientific trajectories unchanged."""

import pytest
from experiments.paper.analysis.records import ScenarioStatistics

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from experiments.paper.pathway_index import plot_trajectories as plotting


def test_negative_trajectory_values_are_preserved_with_fixed_paper_viewport(monkeypatch):
    points = np.array([[-.7, .2], [-.3, .8]])
    image = plotting._density_image([(points, .4)], plt.get_cmap("coolwarm"), Normalize(-1, 1), (-.7, 1.0), bins=8)
    assert np.count_nonzero(image[..., 3]) == 2
    assert image[:, :3, 3].sum() > 0

    record = ScenarioStatistics(name="s_grad_negative", p_index=points[:, 0], h_index=points[:, 1],
                             pathway_integral=.4, repost_rate=plotting.retweet_groups[0],
                             recsys_type=plotting.rs_groups[0][0], post_retain_count=plotting.rs_groups[0][1])
    monkeypatch.setattr(plotting, "summary_curves_clustered", lambda curves: [(p, g, 1) for p, g in curves])
    figure = plotting.plot_p_h_index_trajectories([record])
    try:
        assert all(axis.get_xlim() == (-0.05, 1.05) for axis in figure.axes[:16])
        assert all(axis.get_ylim() == (-0.05, 1.05) for axis in figure.axes[:16])
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), points[:, 0])
        np.testing.assert_array_equal(record.p_index, points[:, 0])
        np.testing.assert_allclose(figure.axes[-1].get_ylim(), (0., 1.))
        np.testing.assert_allclose(figure.axes[-1].get_yticks(), np.linspace(0., 1., 6))
        assert all("n=" not in axis.get_title(loc="left") for axis in figure.axes[:16])
    finally:
        plt.close(figure)


def test_plotting_rejects_mismatched_scientific_trajectories():
    with pytest.raises(ValueError,match="equal-length"):
        plotting._safe_curve(np.array([0.,.2,.4]),np.array([0.,1.]))


def test_trajectory_counts_are_written_to_caption(tmp_path):
    record = ScenarioStatistics(name="s_grad_example", p_index=np.array([0., 0.2]), h_index=np.array([0., 1.]),
                                pathway_integral=0.4, repost_rate=0., recsys_type="StructureM9", post_retain_count=0)
    plotting.write_trajectory_caption(tmp_path / "figure", [record])
    caption = (tmp_path / "figure_caption.txt").read_text(encoding="utf-8")
    assert "StructureM9/k=0: n=1" in caption and "point-index grid" in caption
