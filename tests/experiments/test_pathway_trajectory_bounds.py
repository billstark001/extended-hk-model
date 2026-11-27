"""Negative canonical L values remain visible in densities and trajectory axes."""

import pytest
from experiments.paper.analysis.records import ScenarioStatistics

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from experiments.paper.pathway_index import plot_trajectories as plotting


def test_negative_trajectory_values_contribute_to_density_and_shared_extent(monkeypatch):
    points = np.array([[-.7, .2], [-.3, .8]])
    image = plotting._density_image([(points, .4)], plt.get_cmap("coolwarm"), Normalize(-1, 1), (-.7, 1.0), bins=8)
    assert np.count_nonzero(image[..., 3]) == 2
    assert image[:, :3, 3].sum() > 0

    record = ScenarioStatistics(name="s_grad_negative", p_index=points[:, 0], h_index=points[:, 1],
                             pathway_integral=.4, repost_rate=plotting.retweet_groups[0],
                             recsys_type=plotting.rs_groups[0][0], post_retain_count=plotting.rs_groups[0][1])
    monkeypatch.setattr(plotting, "read_statistics", lambda *args, **kwargs: [record])
    monkeypatch.setattr(plotting, "summary_curves_clustered", lambda curves: [(p, g, 1) for p, g in curves])
    figure = plotting.plot_p_h_index_trajectories()
    try:
        assert figure.axes[0].images[0].get_extent()[0] == -.7
        assert all(axis.get_xlim()[0] <= -.7 for axis in figure.axes[:16])
    finally:
        plt.close(figure)


def test_plotting_rejects_mismatched_scientific_trajectories():
    with pytest.raises(ValueError,match="equal-length"):
        plotting._safe_curve(np.array([0.,.2,.4]),np.array([0.,1.]))
