"""Contracts for the sole clustered-reference JS-distance indices."""

import networkx as nx
import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import rel_entr
from scipy.stats import norm

from ehk.metrics import polarization
from ehk.metrics.density import FittedDensity
from ehk.metrics.polarization import DistanceCalculator
from ehk.metrics.state import MeasurementState


def _fixture(dtype=np.float64):
    n = 17
    opinions = np.random.default_rng(400).uniform(-1, 1, n).astype(dtype)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((a, b) for a in range(n) for b in range(n) if a != b and (a + 3 * b) % 5 < 2)
    return graph, opinions


def _distance(p, q, axis, floor):
    p = np.maximum(p / trapezoid(p, axis), floor)
    q = np.maximum(q / trapezoid(q, axis), floor)
    m = (p + q) / 2
    return np.sqrt(trapezoid(.5 * (rel_entr(p, m) + rel_entr(q, m)), axis))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_indices_use_normalized_clustered_reference_distance(dtype):
    graph, opinions = _fixture(dtype)
    calculator = DistanceCalculator(sample_count=17, max_kde_samples=37)
    state = MeasurementState.from_graph(graph, opinions)
    opinion = calculator.prepare_opinion(opinions, .45, random_state=83)
    edge = calculator.prepare_edge(state, random_state=83)
    expected = []
    for density, clustered in (
        (opinion.density, opinion.clustered_reference),
        (edge, calculator.clustered_edge_reference),
    ):
        observed = density.pdf(calculator.axis)
        expected.append(1 - _distance(observed, clustered, calculator.axis, calculator.t_err)
                        / _distance(clustered, calculator.random_reference, calculator.axis, calculator.t_err))
    actual = calculator.calculate_indices(state, .45, random_state=83, method="exact")
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    assert calculator.calculate_indices(state, .45, random_state=np.random.default_rng(83), method="exact") == actual
    assert (calculator.opinion_index(opinion, method="exact"), calculator.edge_index(edge, method="exact")) == actual


def test_clustered_reference_anchors_negative_values_and_density_normalization():
    calculator = DistanceCalculator(sample_count=17)

    def fitted(values):
        return FittedDensity(np.array([0.]), lambda axis: values, .01, True)

    random = calculator.random_reference
    clustered = calculator.clustered_edge_reference
    assert calculator.edge_index(fitted(random)) == pytest.approx(0., abs=2e-15)
    assert calculator.edge_index(fitted(clustered * 7)) == pytest.approx(1., abs=2e-15)
    opposite = norm.pdf(calculator.axis, 2, .01)
    assert calculator.edge_index(fitted(opposite)) < 0


def test_component_samples_keep_rng_order_without_edge_allpair_work(monkeypatch):
    graph, opinions = _fixture()
    calculator = DistanceCalculator(sample_count=17, max_kde_samples=37)
    state = MeasurementState.from_graph(graph, opinions)
    rng = np.random.default_rng(83)
    i, j = np.triu_indices(len(opinions), k=1)
    pairs = np.abs(opinions[i] - opinions[j])
    expected_opinion = pairs[rng.choice(len(pairs), 37, replace=False)]
    expected_edge = state.edge_distances[rng.choice(len(state.edges), 37, replace=False)]
    objective = calculator.prepare_opinion(opinions, .45, random_state=83)
    np.testing.assert_array_equal(objective.density.samples, expected_opinion)
    assert objective.density.bandwidth == objective.density.pdf.cho_cov[0, 0]

    def no_allpair(*args, **kwargs):
        pytest.fail("preparing edge components must not recompute all-pair differences")

    monkeypatch.setattr(calculator, "_prepare_opinion", no_allpair)
    monkeypatch.setattr(np, "triu_indices", no_allpair)
    edge = calculator.prepare_edge(state, random_state=83)
    np.testing.assert_array_equal(edge.samples, expected_edge)
    assert edge.bandwidth == edge.pdf.cho_cov[0, 0]


def test_binned_is_the_calculator_default_and_all_supported_resolutions_share_the_formula():
    graph, opinions = _fixture()
    calculator = DistanceCalculator(sample_count=17, max_kde_samples=37)
    state = MeasurementState.from_graph(graph, opinions)
    assert calculator.density_method == "binned" and calculator.bins == 4081
    actual = calculator.calculate_indices(state, .45, random_state=31)
    assert actual == calculator.calculate_indices(state, .45, random_state=31, method="binned", bins=4081)
    exact = calculator.calculate_indices(state, .45, random_state=31, method="exact")
    for bins in (1021, 2041, 4081, 8161):
        approximate = calculator.calculate_indices(state, .45, random_state=31, bins=bins)
        np.testing.assert_allclose(approximate, exact, atol=3e-4, rtol=0)


def test_binned_components_do_not_evaluate_exact_observed_pdfs(monkeypatch):
    graph, opinions = _fixture()
    calculator = DistanceCalculator(sample_count=17, max_kde_samples=37)
    fit = polarization.get_kde_pdf

    class FitOnly:
        def __init__(self, pdf):
            self.cho_cov = pdf.cho_cov

        def __call__(self, axis):
            pytest.fail("binned nonconstant measurements must not evaluate exact observed PDFs")

    monkeypatch.setattr(polarization, "get_kde_pdf", lambda *args: FitOnly(fit(*args)))
    values = calculator.calculate_indices(MeasurementState.from_graph(graph, opinions), .45, random_state=83)
    assert np.isfinite(values).all()


def test_constant_samples_use_analytic_gaussian_without_fft(monkeypatch):
    from ehk.metrics import density

    graph, _ = _fixture()
    calculator = DistanceCalculator(sample_count=17)
    state = MeasurementState.from_graph(graph, np.zeros(17))
    expected = calculator.calculate_indices(state, .45, random_state=83, method="exact")

    def forbidden(*args, **kwargs):
        pytest.fail("constant samples must use the analytic Gaussian PDF")

    monkeypatch.setattr(density, "linear_binned_gaussian", forbidden)
    assert calculator.calculate_indices(state, .45, random_state=83) == expected
    assert expected[1] == 1.
    assert expected[0] != 0 and expected[0] <= 1.


def test_parameter_key_tracks_mutated_parameters_and_actual_references():
    calculator = DistanceCalculator(sample_count=17)
    old = calculator.parameter_key()
    calculator.random_reference[0] += .01
    assert calculator.parameter_key() != old
    old = calculator.parameter_key()
    calculator.max_kde_samples = 19
    assert calculator.parameter_key() != old


@pytest.mark.parametrize("options", [{"density_method": "unknown"}, {"bins": 4096}])
def test_invalid_backend_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        DistanceCalculator(sample_count=17, **options)
