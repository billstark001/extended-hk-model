"""Numerical conventions for confidence and activity metrics."""

import networkx as nx
import numpy as np
import pytest

from ehk.metrics.homophily import mean_followee_edge_concordance
from ehk.micro.observables import calc_active_step, get_gradation_index_hp


def test_edge_threshold_and_signed_path_integral_follow_paper_definition():
    graph = nx.DiGraph([(0, 1)])
    assert mean_followee_edge_concordance(graph, np.array([0.0, 0.45]), 0.45) == 0
    assert get_gradation_index_hp(np.array([0, 0.5, 0.1]), np.array([0, 0, 1])) == pytest.approx(-0.2)
    # No terminal extension to I_p=1.
    assert get_gradation_index_hp(np.array([0, 0.2]), np.array([0.5, 0.5])) == pytest.approx(0.1)

def test_activity_uses_terminal_reference_and_falls_back_if_never_crossed():
    steps = np.array([0, 10, 20])
    result = calc_active_step(steps, np.array([0.0, 1.0, 0.8]), 0.98, 0.75)
    assert result[1] == 7  # max(.98*.8,.75)=.784; historical max-over-time would give 9.
    absent = calc_active_step(steps, np.array([0.1, 0.2, 0.3]), 0.98, 0.75)
    assert absent[1] == 20 and absent[4] == pytest.approx(0.2)

def test_activity_already_above_criterion_starts_at_initial_observation():
    result = calc_active_step(np.array([0, 10, 20]), np.array([1.0, 1.0, 1.0]), 0.98, 0.75)
    assert result[:3] == (0, 0, 1.0)
    np.testing.assert_array_equal(result[3], [1.0])
    assert result[4] == 1.0
