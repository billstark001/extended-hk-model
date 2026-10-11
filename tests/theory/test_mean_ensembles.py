"""Statistical units and whole-curve clocks, independent of experiment data."""

import numpy as np
import pytest

from common.statistics import bootstrap_weights, hierarchical_weights
from theory.release import release_clock
from theory.release.ensemble import clock_statistics, plateau_check

PROTOCOL = {"bootstrap_seed": 5, "bootstrap_samples": 200, "confidence_level": 0.95}


def test_fixed_plateau_integral_and_paired_ratio():
    time = np.linspace(0, 10, 1001)
    node = np.array([np.exp(-time), np.exp(-time)])
    edge = np.array([np.exp(-2 * time), np.exp(-2 * time)])
    stat, samples = clock_statistics(time, node, edge, bootstrap_weights(2, PROTOCOL), plateaus={"node": 0, "edge": 0})
    assert stat["node_mean"] == pytest.approx(1 - np.exp(-10), rel=1e-4)
    assert stat["edge_mean"] == pytest.approx(0.5 * (1 - np.exp(-20)), rel=1e-4)
    assert stat["edge_to_node_mean"] == pytest.approx(0.5, rel=1e-4)
    assert np.ptp(samples["edge_to_node"]) == 0
    assert stat["plateau_reference"] == "fixed_pair_endpoint"


def test_bootstrap_estimates_clock_of_mean_curve():
    time = np.linspace(0, 10, 501)
    curves = np.array([np.exp(-time), 3 * np.exp(-time / 3)])
    stat, _ = clock_statistics(time, curves, curves, bootstrap_weights(2, PROTOCOL), plateaus={"node": 0, "edge": 0})
    expected = release_clock(time, curves.mean(0), plateau=0).area
    assert stat["node_mean"] == expected
    individual = np.mean([release_clock(time, c, plateau=0).area for c in curves])
    assert abs(expected - individual) > 0.3


def test_hierarchical_weights_equalize_initial_states_with_unequal_branches():
    weights = hierarchical_weights([1, 4], PROTOCOL)
    assert np.allclose(weights.sum(1), 1)
    assert weights[:, 0].mean() == pytest.approx(0.5, abs=0.08)
    assert weights[:, 1:].sum(1).mean() == pytest.approx(0.5, abs=0.08)
    with pytest.raises(ValueError):
        hierarchical_weights([0], PROTOCOL)


def test_plateau_needs_two_windows_and_long_enough_trace():
    time = np.linspace(0, 200, 2001)
    assert plateau_check(time, np.exp(-time))
    assert not plateau_check(time, time / 200)
    assert not plateau_check(time[:20], np.ones(20))


def test_weak_release_and_single_branch_have_no_invented_interval():
    time = np.linspace(0, 2, 21)
    curves = np.ones((1, 21))
    stat, _ = clock_statistics(time, curves, curves, bootstrap_weights(1, PROTOCOL))
    assert stat["node_status"] == "weak_release"
    assert stat["node_valid_fraction"] == 0
    assert stat["node_lower"] is None
