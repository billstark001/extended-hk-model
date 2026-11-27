"""Shared-state extraction and the Gaussian convolution's numerical boundary."""

import networkx as nx
import numpy as np
import pytest
from scipy.stats import norm

from ehk.metrics.density import linear_binned_gaussian
from ehk.metrics.state import MeasurementState


def test_state_preserves_order_dtype_and_single_extraction_without_mutating_inputs():
    graph = nx.DiGraph([(2, 0), (0, 1), (2, 1)])
    opinions = np.array([-.2, .3, .8, -.7], dtype=np.float32)
    state = MeasurementState.from_graph(graph, opinions)
    expected = np.asarray(list(graph.edges), dtype=int)
    np.testing.assert_array_equal(state.edges, expected)
    np.testing.assert_array_equal(state.edge_distances, np.abs(opinions[expected[:, 1]] - opinions[expected[:, 0]]))
    np.testing.assert_array_equal(state.out_degree, [1, 0, 2, 0])
    assert state.opinions.dtype == state.edge_distances.dtype == np.dtype(np.float32)
    original = state.opinions.copy()
    opinions[:] = 0
    np.testing.assert_array_equal(state.opinions, original)
    for array in (state.opinions, state.edges, state.edge_distances, state.out_degree):
        assert not array.flags.writeable
    array_state = MeasurementState.from_edges(original, expected)
    for name in ("opinions", "edges", "edge_distances", "out_degree"):
        np.testing.assert_array_equal(getattr(state, name), getattr(array_state, name))


def test_empty_topology_and_invalid_endpoints():
    state = MeasurementState.from_edges(np.array([-.1, .1]), [])
    assert state.edges.shape == (0, 2)
    assert state.edge_distances.shape == (0,)
    np.testing.assert_array_equal(state.out_degree, [0, 0])
    with pytest.raises(ValueError, match="endpoints"):
        MeasurementState.from_edges(np.zeros(2), [[0, 2]])
    with pytest.raises(ValueError, match="integer"):
        MeasurementState.from_edges(np.zeros(2), [[0., 1.]])


def test_zero_padding_has_no_boundary_wrap_and_original_axis_is_embedded():
    bins = 4081
    axis = np.linspace(-.04, 2.04, 256)
    fine = np.linspace(axis[0], axis[-1], bins)
    samples = fine[[0, 17, bins // 3, bins - 1]]
    bandwidth = .075
    expected = norm.pdf((axis[:, None] - samples) / bandwidth).mean(axis=1) / bandwidth
    actual = linear_binned_gaussian(samples, axis, bandwidth)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    assert actual.dtype == np.float64
    assert actual.shape == axis.shape
    assert np.all(actual >= 0)


def test_linear_binning_respects_gaussian_interpolation_error_bound():
    axis = np.linspace(-.04, 2.04, 256)
    samples = np.array([.013, .279, .703, 1.117, 1.483, 1.987])
    bandwidth = .01
    exact = norm.pdf((axis[:, None] - samples) / bandwidth).mean(axis=1) / bandwidth
    actual = linear_binned_gaussian(samples, axis, bandwidth)
    spacing = (axis[-1] - axis[0]) / 4080
    bound = spacing**2 / (8 * np.sqrt(2 * np.pi) * bandwidth**3)
    assert np.max(np.abs(actual - exact)) <= bound + 2e-12


def test_opt_in_fft_diagnostics_leave_the_array_api_and_values_unchanged():
    axis = np.linspace(-.04, 2.04, 256)
    samples = np.array([.013, .279, .703, 1.117, 1.483, 1.987])
    expected = linear_binned_gaussian(samples, axis, .01)
    diagnostics = {}
    actual = linear_binned_gaussian(samples, axis, .01, diagnostics=diagnostics)
    np.testing.assert_array_equal(actual, expected)
    assert set(diagnostics) == {"negative_count", "minimum_before_clip", "fft_length"}
    assert isinstance(diagnostics["negative_count"], int)
    assert isinstance(diagnostics["minimum_before_clip"], float)
    assert diagnostics["fft_length"] >= 3 * 4081 - 2


@pytest.mark.parametrize("bins", [0, True, 1024])
def test_nonembedded_grid_is_rejected(bins):
    with pytest.raises(ValueError, match="255"):
        linear_binned_gaussian(np.array([0., 1.]), np.linspace(-.04, 2.04, 256), .01, bins)
