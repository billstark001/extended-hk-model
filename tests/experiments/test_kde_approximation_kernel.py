"""The B experiment changes only observed Gaussian-density evaluation."""

import json
import subprocess

import networkx as nx
import numpy as np
import pytest
from scipy.stats import norm

from ehk.metrics import polarization
from ehk.metrics.density import linear_binned_gaussian
from ehk.metrics.polarization import DistanceCalculator, ideal_dist_worst_obj_array
from experiments.paper.kde_approximation import kernel


@pytest.fixture(autouse=True)
def no_native_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("KDE kernel tests must not run a simulator or a probe")

    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def reference():
    return DistanceCalculator(use_js_divergence=True, min_bandwidth=.01, sample_count=12, max_kde_samples=37)


def _graph(n=12):
    return nx.complete_graph(n, create_using=nx.DiGraph)


def test_default_reuses_the_existing_calculator():
    from ehk.micro.observables import distance_collector

    calculator = kernel.BinnedDistanceCalculator()
    assert calculator.reference is distance_collector
    assert np.array_equal(calculator.axis, distance_collector.axis)


def test_prepare_matches_original_sampling_rng_order_bandwidth_and_exact_values(monkeypatch):
    from ehk.micro.observables import distance_collector

    # Both populations exceed the original 10,000 cap. The edge sample must
    # consume the same RNG after the all-pair sample, not a newly seeded RNG.
    graph = _graph(145)
    opinions = np.random.default_rng(29).uniform(-1, 1, 145)
    captured, fitted = [], []
    original_fit = polarization.get_kde_pdf

    def fit(samples, *args):
        captured.append(np.array(samples, copy=True))
        result = original_fit(samples, *args)
        fitted.append(result)
        return result

    monkeypatch.setattr(polarization, "get_kde_pdf", fit)
    expected = distance_collector.calculate(graph, opinions, .45, random_state=31)
    calculator = kernel.BinnedDistanceCalculator(distance_collector)
    prepared = calculator.prepare(graph, opinions, .45, 31)
    assert [len(values) for values in captured] == [10000] * 4
    np.testing.assert_array_equal(captured[0], captured[2])
    np.testing.assert_array_equal(captured[1], captured[3])
    assert prepared.opinion.bandwidth == fitted[0].cho_cov[0, 0]
    assert prepared.edge.bandwidth == fitted[1].cho_cov[0, 0]
    ip, isub, diagnostics = calculator.exact(prepared)
    assert (ip, isub) == (expected.rand_o, expected.rand_s)
    assert diagnostics["method"] == "exact"


def test_objective_reference_uses_complete_pairs_not_the_kde_subsample(reference):
    opinions = np.linspace(-1, 1, 12)
    calculator = kernel.BinnedDistanceCalculator(reference)
    prepared = calculator.prepare(_graph(), opinions, .45, 7)
    i, j = np.triu_indices(12, k=1)
    pairs = np.abs(opinions[i] - opinions[j])
    expected = float(np.mean(pairs[pairs >= .45]))
    assert prepared.objective_separation == expected
    subset = prepared.opinion.samples
    assert float(np.mean(subset[subset >= .45])) != expected
    np.testing.assert_array_equal(prepared.objective_reference, ideal_dist_worst_obj_array(
        reference.axis, expected, 37, reference.min_bandwidth,
    ))
    assert not prepared.opinion.samples.flags.writeable
    assert not prepared.objective_reference.flags.writeable


def test_experiment_selects_resolutions_and_records_canonical_component_diagnostics(reference, monkeypatch):
    calls = []
    for name in ("prepare_opinion", "prepare_edge", "opinion_index", "edge_index"):
        original = getattr(reference, name)

        def observed(*args, _name=name, _original=original, **kwargs):
            calls.append((_name, kwargs.get("method"), kwargs.get("bins")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(reference, name, observed)
    calculator = kernel.BinnedDistanceCalculator(reference)
    prepared = calculator.prepare(_graph(), np.linspace(-1, 1, 12), .45, 0)
    assert prepared.opinion is prepared.objective.density
    assert prepared.objective_reference is prepared.objective.objective_reference
    assert prepared.objective_separation == prepared.objective.objective_separation
    calculator.exact(prepared)
    for bins in kernel.SUPPORTED_BINS:
        _, _, diagnostics = calculator.evaluate(prepared, bins)
        assert diagnostics["opinion_fft_length"] >= 3 * bins - 2
        assert diagnostics["edge_fft_length"] >= 3 * bins - 2
    assert calls[:2] == [("prepare_opinion", None, None), ("prepare_edge", None, None)]
    assert calls[2:4] == [("opinion_index", "exact", None), ("edge_index", "exact", None)]
    assert calls[4:] == [(name, "binned", bins) for bins in kernel.SUPPORTED_BINS for name in ("opinion_index", "edge_index")]


@pytest.mark.parametrize("bins", kernel.SUPPORTED_BINS)
def test_aligned_mass_matches_direct_gaussians_without_boundary_wrap(bins):
    axis = np.linspace(-.04, 2.04, 256)
    fine = np.linspace(axis[0], axis[-1], bins)
    samples = fine[[0, 17, bins // 3, bins - 1]]
    bandwidth = .075
    expected = norm.pdf((axis[:, None] - samples) / bandwidth).mean(axis=1) / bandwidth
    diagnostics = {}
    actual = linear_binned_gaussian(samples, axis, bandwidth, bins, diagnostics=diagnostics)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    assert actual.shape == axis.shape
    assert diagnostics["fft_length"] >= 3 * bins - 2
    assert diagnostics["minimum_before_clip"] >= -2e-12
    assert np.all(actual >= 0)


def test_off_grid_density_respects_linear_binning_error_bound():
    axis = np.linspace(-.04, 2.04, 256)
    samples = np.array([.013, .279, .703, 1.117, 1.483, 1.987])
    bandwidth = .025
    exact = norm.pdf((axis[:, None] - samples) / bandwidth).mean(axis=1) / bandwidth
    errors = []
    for bins in kernel.SUPPORTED_BINS:
        actual = linear_binned_gaussian(samples, axis, bandwidth, bins)
        spacing = (axis[-1] - axis[0]) / (bins - 1)
        bound = spacing**2 / (8 * np.sqrt(2 * np.pi) * bandwidth**3)
        error = float(np.max(np.abs(actual - exact)))
        assert error <= bound + 2e-12
        errors.append(error)
    assert errors[-1] < errors[0] / 8


def test_constant_samples_keep_original_fallback_for_every_resolution(reference):
    graph, opinions = _graph(), np.zeros(12)
    calculator = kernel.BinnedDistanceCalculator(reference)
    prepared = calculator.prepare(graph, opinions, .45, 0)
    expected = reference.calculate(graph, opinions, .45, random_state=0)
    for bins in kernel.SUPPORTED_BINS:
        ip, isub, diagnostics = calculator.evaluate(prepared, bins)
        assert (ip, isub) == (expected.rand_o, expected.rand_s)
        assert diagnostics["opinion_constant"] and diagnostics["edge_constant"]
        assert diagnostics["opinion_fft_length"] == diagnostics["edge_fft_length"] == 0
        assert diagnostics["stride"] == (bins - 1) // 255
        json.dumps(diagnostics, allow_nan=False)


def test_nonconstant_evaluation_preserves_inputs_and_returns_small_json_diagnostics(reference):
    calculator = kernel.BinnedDistanceCalculator(reference)
    prepared = calculator.prepare(_graph(), np.linspace(-.91, .97, 12), .45, 13)
    snapshots = [values.copy() for values in (
        prepared.opinion.samples, prepared.edge.samples, prepared.objective_reference,
    )]
    exact = calculator.exact(prepared)
    errors = []
    for bins in kernel.SUPPORTED_BINS:
        ip, isub, diagnostics = calculator.evaluate(prepared, bins)
        assert np.isfinite([ip, isub]).all()
        assert len(json.dumps(diagnostics, allow_nan=False)) < 1500
        errors.append(max(abs(ip - exact[0]), abs(isub - exact[1])))
    assert errors[-1] < errors[0]
    for values, before in zip((prepared.opinion.samples, prepared.edge.samples, prepared.objective_reference), snapshots):
        np.testing.assert_array_equal(values, before)


def test_changed_bound_calculator_or_foreign_preparation_is_rejected(reference, monkeypatch):
    first, second = kernel.BinnedDistanceCalculator(reference), kernel.BinnedDistanceCalculator(reference)
    prepared = first.prepare(_graph(), np.linspace(-1, 1, 12), .45, 0)
    with pytest.raises(ValueError, match="different calculator"):
        second.evaluate(prepared, 1021)
    monkeypatch.setattr(reference, "min_bandwidth", .02)
    with pytest.raises(ValueError, match="changed"):
        first.exact(prepared)


@pytest.mark.parametrize("bins", [0, 256, 1024, True])
def test_other_experimental_resolutions_are_rejected(bins, reference):
    calculator = kernel.BinnedDistanceCalculator(reference)
    prepared = calculator.prepare(_graph(), np.linspace(-1, 1, 12), .45, 0)
    with pytest.raises(ValueError, match="bins must"):
        calculator.evaluate(prepared, bins)
