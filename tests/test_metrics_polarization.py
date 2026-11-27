"""Numerical contracts for the production distance components and exact archive."""

import networkx as nx
import numpy as np
import pytest

from ehk.metrics import polarization
from ehk.metrics.polarization import DistanceCalculator
from ehk.metrics.state import MeasurementState


def _fixture(dtype=np.float64):
    n = 17
    opinions = np.random.default_rng(400).uniform(-1, 1, n).astype(dtype)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((a, b) for a in range(n) for b in range(n) if a != b and (a + 3 * b) % 5 < 2)
    return graph, opinions


# Captured from the original exact implementation before extracting components.
# Both populations are subsampled, and KL keeps its historical scale directions.
EXACT_HEX = {
    ("float64", False): ["-0x1.4358d05f29ac4p+0", "-0x1.133afeef3f266p-7", "0x1.ee5d9e144fe92p-1", "0x1.dfcc25c24a7cap-1"],
    ("float64", True): ["0x1.03cb23e4cf9d9p-5", "0x1.5d29d2216cb66p-8", "0x1.f6a44b47c9d71p-1", "0x1.f57e09f4286cfp-1"],
    ("float32", False): ["-0x1.4358d4a0f1dd3p+0", "-0x1.133afe79fbe0ep-7", "0x1.ee5d9dcb6de12p-1", "0x1.dfcc25c08e745p-1"],
    ("float32", True): ["0x1.03cb225353d79p-5", "0x1.5d29d6ab75313p-8", "0x1.f6a44b56a65bdp-1", "0x1.f57e09f729164p-1"],
}


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("js", [False, True])
def test_original_exact_results_are_bitwise_preserved(dtype, js):
    graph, opinions = _fixture(dtype)
    calculator = DistanceCalculator(
        min_bandwidth=.01, sample_count=17, max_kde_samples=37, use_js_divergence=js, use_debug_data=True,
    )
    result = calculator.calculate(graph, opinions, .45, random_state=83)
    assert calculator.density_method == "exact"
    assert [float(getattr(result, name)).hex() for name in ("rand_o", "rand_s", "worst_o", "worst_s")] == EXACT_HEX[(dtype.__name__, js)]
    state = MeasurementState.from_graph(graph, opinions)
    assert calculator.calculate_indices(state, .45, random_state=83) == (result.rand_o, result.rand_s)
    assert calculator.calculate_indices(state, .45, random_state=np.random.default_rng(83)) == (result.rand_o, result.rand_s)
    objective = calculator.prepare_opinion(opinions, .45, random_state=83)
    edge = calculator.prepare_edge(state, random_state=83)
    assert (calculator.opinion_index(objective), calculator.edge_index(edge)) == (result.rand_o, result.rand_s)
    np.testing.assert_array_equal(result.debug_data.o_sample, objective.distances)
    np.testing.assert_array_equal(result.debug_data.s_sample, state.edge_distances)


def test_component_samples_keep_original_rng_order_without_edge_allpair_work(monkeypatch):
    graph, opinions = _fixture()
    calculator = DistanceCalculator(min_bandwidth=.01, sample_count=17, max_kde_samples=37, use_js_divergence=True)
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
        pytest.fail("preparing edge components must not recompute pairwise opinion distances")

    monkeypatch.setattr(calculator, "_prepare_opinion", no_allpair)
    monkeypatch.setattr(np, "triu_indices", no_allpair)
    edge = calculator.prepare_edge(state, random_state=83)
    np.testing.assert_array_equal(edge.samples, expected_edge)
    assert edge.bandwidth == edge.pdf.cho_cov[0, 0]


@pytest.mark.parametrize("n,maximum,expected", [
    (17, 37, ["0x1.575ef60338609p-3", "0x1.1364ce8075b10p-8"]),
    (145, 10000, ["0x1.fe0bae63e48a5p-10", "0x1.20d7fa7ee04f6p-10"]),
])
def test_production_matches_b4081_values_captured_before_promotion(n, maximum, expected):
    graph = nx.complete_graph(n, create_using=nx.DiGraph)
    opinions = np.random.default_rng(29).uniform(-1, 1, n)
    reference = DistanceCalculator(use_js_divergence=True, min_bandwidth=.01, sample_count=n, max_kde_samples=maximum)
    state = MeasurementState.from_edges(opinions, np.asarray(list(graph.edges), dtype=int))
    values = reference.calculate_indices(state, .45, random_state=31, method="binned", bins=4081)
    assert [float(value).hex() for value in values] == expected
    assert reference.density_method == "exact"


def test_binned_components_do_not_evaluate_exact_observed_pdfs(monkeypatch):
    graph, opinions = _fixture()
    calculator = DistanceCalculator(use_js_divergence=True, min_bandwidth=.01, sample_count=17, max_kde_samples=37)
    fit = polarization.get_kde_pdf

    class FitOnly:
        def __init__(self, pdf):
            self.cho_cov = pdf.cho_cov

        def __call__(self, axis):
            pytest.fail("binned nonconstant measurements must not evaluate exact observed PDFs")

    monkeypatch.setattr(polarization, "get_kde_pdf", lambda *args: FitOnly(fit(*args)))
    values = calculator.calculate_indices(MeasurementState.from_graph(graph, opinions), .45, random_state=83, method="binned")
    assert np.isfinite(values).all()


def test_archived_worst_components_remain_exact_under_explicit_binned_indices():
    graph, opinions = _fixture()
    calculator = DistanceCalculator(use_js_divergence=True, min_bandwidth=.01, sample_count=17, max_kde_samples=37)
    exact = calculator.calculate(graph, opinions, .45, random_state=83)
    binned = calculator.calculate(graph, opinions, .45, random_state=83, method="binned", bins=4081)
    assert (binned.worst_o, binned.worst_s) == (exact.worst_o, exact.worst_s)
    assert (binned.rand_o, binned.rand_s) == calculator.calculate_indices(
        MeasurementState.from_graph(graph, opinions), .45, random_state=83, method="binned",
    )


def test_constant_samples_keep_the_original_gaussian_and_no_fft(monkeypatch):
    from ehk.metrics import density

    graph, _ = _fixture()
    opinions = np.zeros(17)
    calculator = DistanceCalculator(use_js_divergence=True, min_bandwidth=.01, sample_count=17)
    state = MeasurementState.from_graph(graph, opinions)
    expected = calculator.calculate_indices(state, .45, random_state=83)

    def forbidden(*args, **kwargs):
        pytest.fail("constant samples must use the original Gaussian PDF")

    monkeypatch.setattr(density, "linear_binned_gaussian", forbidden)
    assert calculator.calculate_indices(state, .45, random_state=83, method="binned") == expected


def test_parameter_key_tracks_mutated_parameters_and_actual_references():
    calculator = DistanceCalculator(use_js_divergence=True, min_bandwidth=.01, sample_count=17)
    old = calculator.parameter_key()
    calculator.o_rand_vals[0] += .01
    assert calculator.parameter_key() != old
    old = calculator.parameter_key()
    calculator.max_kde_samples = 19
    assert calculator.parameter_key() != old


@pytest.mark.parametrize("options", [{"density_method": "unknown"}, {"density_method": "binned"}, {"bins": 4096}])
def test_invalid_backend_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        DistanceCalculator(sample_count=17, **options)
