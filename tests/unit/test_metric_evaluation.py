"""Production caches follow physical and numerical dependencies independently."""

import gc
import weakref

import networkx as nx
import numpy as np
import pytest

from ehk.metrics.evaluation import MeasurementEvaluator
from ehk.metrics.homophily import mean_followee_edge_concordance, normalize_homophily
from ehk.metrics.polarization import DistanceCalculator
from ehk.metrics.state import MeasurementRecord, MeasurementState


class Record(MeasurementRecord):
    def __init__(self, graph=None, opinions=None, *, epsilon=.45, horizon=4):
        self.graph = graph if graph is not None else nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (3, 0)])
        values = np.array([-.75, -.25, .25, .75]) if opinions is None else np.asarray(opinions)
        super().__init__(opinions=np.tile(values, (horizon + 1, 1)), max_step=horizon,
                         epsilon=epsilon, graph_at=self.get_graph)
        self.graph_calls = 0

    def get_graph(self, step):
        self.graph_calls += 1
        return self.graph


class Calculator:
    def __init__(self):
        self.opinion_calls = self.edge_calls = 0
        self.factor = 1.

    def parameter_key(self):
        return (self.factor,)

    def prepare_opinion(self, opinions, t_opinion, *, random_state):
        self.opinion_calls += 1
        return float(opinions.sum()) + t_opinion + random_state / 100

    def opinion_index(self, prepared, *, method, bins):
        return prepared * self.factor

    def prepare_edge(self, state, *, random_state):
        self.edge_calls += 1
        return float(state.edge_distances.sum()) + random_state / 100

    def edge_index(self, prepared, *, method, bins):
        return prepared * self.factor


def test_point_extracts_graph_once_and_reuses_opinion_cache_across_graphs():
    calculator = Calculator()
    evaluator = MeasurementEvaluator(calculator=calculator)
    first, second = Record(), Record()
    second.graph.add_edge(1, 3)
    p0, s0, h0 = evaluator.point(first, 1)
    p1, s1, h1 = evaluator.point(second, 1)
    assert first.graph_calls == second.graph_calls == 1
    assert p0 == p1 and s0 != s1
    assert calculator.opinion_calls == 1 and calculator.edge_calls == 2
    assert h0 == normalize_homophily(mean_followee_edge_concordance(first.graph, first.opinions[1], .45), .45)
    assert h1 == normalize_homophily(mean_followee_edge_concordance(second.graph, second.opinions[1], .45), .45)
    assert evaluator.cache_info().opinion.hits == 1
    assert evaluator.cache_info().homophily.misses == 2


def test_homophily_only_does_not_prepare_any_density_and_can_reuse_static_steps():
    calculator = Calculator()
    evaluator = MeasurementEvaluator(calculator=calculator)
    record = Record()
    first = evaluator.homophily(record, 0)
    assert evaluator.homophily(record, 4) == first
    assert calculator.opinion_calls == calculator.edge_calls == 0
    assert evaluator.cache_info().homophily.hits == 1
    evaluator.distance(record, 0)
    assert calculator.opinion_calls == calculator.edge_calls == 1


def test_homophily_only_does_not_construct_a_reference_calculator(monkeypatch):
    from ehk.metrics import polarization

    def forbidden(*args, **kwargs):
        pytest.fail("homophily-only lookup created a KDE reference")

    monkeypatch.setattr(polarization, "DistanceCalculator", forbidden)
    assert np.isfinite(MeasurementEvaluator().homophily(Record(), 0))


def test_float32_confidence_boundary_uses_the_original_float64_homophily_subtraction():
    opinions = np.array([.010000006, .46], dtype=np.float32)
    # A float32 subtraction rounds below .45; promoting each operand first
    # gives a distance above .45, which the original homophily rule rejects.
    assert float(opinions[1] - opinions[0]) < .45
    assert float(opinions[1]) - float(opinions[0]) > .45
    record = Record(nx.DiGraph([(0, 1)]), opinions, epsilon=.45)
    assert MeasurementEvaluator().homophily(record, 0) == 0.


@pytest.mark.parametrize("change", ["step", "precision", "opinion_order", "opinion_mutation"])
def test_opinion_dependency_changes_invalidate_both_distance_components(change):
    calculator = Calculator()
    evaluator = MeasurementEvaluator(calculator=calculator)
    first, second = Record(), Record()
    evaluator.distance(first, 1)
    step = 1
    if change == "step":
        step = 2
    elif change == "precision":
        second.opinions = second.opinions.astype(np.float32)
    elif change == "opinion_order":
        second.opinions[step] = second.opinions[step, ::-1]
    else:
        second.opinions[step, 0] = np.nextafter(second.opinions[step, 0], 0)
    evaluator.distance(second, step)
    assert calculator.opinion_calls == calculator.edge_calls == 2


def test_confidence_change_invalidates_opinion_and_homophily_but_reuses_edge_index():
    calculator = Calculator()
    evaluator = MeasurementEvaluator(calculator=calculator)
    first, second = Record(epsilon=.45), Record(epsilon=.55)
    p0, s0, h0 = evaluator.point(first, 1)
    p1, s1, h1 = evaluator.point(second, 1)
    assert p0 != p1 and h0 != h1 and s0 == s1
    info = evaluator.cache_info()
    assert info.opinion.misses == info.homophily.misses == 2
    assert info.edge.misses == info.edge.hits == 1
    assert calculator.opinion_calls == 2 and calculator.edge_calls == 1


def test_in_place_graph_and_opinion_mutations_are_detected_without_record_identity_cache():
    calculator = Calculator()
    evaluator = MeasurementEvaluator(calculator=calculator)
    record = Record(epsilon=.55)
    before = evaluator.point(record, 1)
    record.graph.remove_edge(0, 1)
    record.graph.add_edge(0, 3)
    changed = evaluator.point(record, 1)
    assert before[0] == changed[0] and before[1:] != changed[1:]
    assert calculator.opinion_calls == 1 and calculator.edge_calls == 2
    record.opinions[1, 3] = -.7
    again = evaluator.point(record, 1)
    assert calculator.opinion_calls == 2 and calculator.edge_calls == 3
    assert again[2] == normalize_homophily(mean_followee_edge_concordance(record.graph, record.opinions[1], .55), .55)


def test_edge_order_change_is_a_new_edge_sample_but_not_a_new_opinion_sample():
    calculator = Calculator()
    evaluator = MeasurementEvaluator(calculator=calculator)
    record = Record()
    original = list(record.graph.edges)
    evaluator.distance(record, 2)
    record.graph.remove_edge(0, 1)
    record.graph.add_edge(0, 1)
    assert set(original) == set(record.graph.edges) and original != list(record.graph.edges)
    evaluator.distance(record, 2)
    assert calculator.opinion_calls == 1 and calculator.edge_calls == 2


def test_calculator_parameter_change_does_not_reuse_stale_distance_results():
    calculator = Calculator()
    evaluator = MeasurementEvaluator(calculator=calculator)
    record = Record()
    p0, s0 = evaluator.distance(record, 0)
    calculator.factor = 2.
    p1, s1 = evaluator.distance(record, 0)
    assert p1 == 2 * p0 and s1 == 2 * s0
    assert calculator.opinion_calls == calculator.edge_calls == 2


def test_reference_density_mutation_is_part_of_the_distance_cache_identity():
    record = Record()
    calculator = DistanceCalculator(min_bandwidth=.01, sample_count=4)
    evaluator = MeasurementEvaluator(method="exact", calculator=calculator)
    evaluator.distance(record, 1)
    calculator.random_reference[0] *= 1.001
    expected = calculator.calculate_indices(
        MeasurementState.from_graph(record.graph, record.opinions[1]), .45, random_state=1, method="exact",
    )
    actual = evaluator.distance(record, 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    assert evaluator.cache_info().opinion.misses == evaluator.cache_info().edge.misses == 2


def test_each_lru_is_bounded_and_graph_changes_do_not_evict_opinion_entries():
    calculator = Calculator()
    evaluator = MeasurementEvaluator(max_entries=2, calculator=calculator)
    records = [Record() for _ in range(3)]
    records[1].graph.add_edge(1, 3)
    records[2].graph.add_edge(2, 0)
    for record in records:
        evaluator.point(record, 0)
    info = evaluator.cache_info()
    assert info.opinion.size == 1
    assert info.edge.size == info.homophily.size == 2
    evaluator.point(records[0], 0)
    assert calculator.opinion_calls == 1 and calculator.edge_calls == 4
    evaluator.clear()
    assert all(row.size == row.hits == row.misses == 0 for row in evaluator.cache_info().components())


def test_caches_do_not_keep_records_or_graphs_alive():
    evaluator = MeasurementEvaluator(calculator=Calculator())
    record = Record()
    record_ref, graph_ref = weakref.ref(record), weakref.ref(record.graph)
    evaluator.point(record, 0)
    del record
    gc.collect()
    assert record_ref() is None and graph_ref() is None


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_explicit_exact_mode_preserves_old_sampling_with_an_opinion_cache_hit(dtype):
    opinions = np.random.default_rng(21).uniform(-1, 1, 12).astype(dtype)
    first = Record(nx.DiGraph([(i, (i + 1) % 12) for i in range(12)]), opinions, epsilon=.45)
    second = Record(nx.complete_graph(12, create_using=nx.DiGraph), opinions, epsilon=.45)
    calculator = DistanceCalculator(min_bandwidth=.01, sample_count=12, max_kde_samples=37)
    evaluator = MeasurementEvaluator(method="exact", calculator=calculator)
    evaluator.point(first, 3)
    expected = calculator.calculate_indices(
        MeasurementState.from_graph(second.graph, second.opinions[3]), .45, random_state=3, method="exact",
    )
    actual = evaluator.point(second, 3)
    np.testing.assert_allclose(actual[:2], expected, rtol=1e-13, atol=1e-14)
    assert actual[2] == normalize_homophily(mean_followee_edge_concordance(second.graph, second.opinions[3], .45), .45)
    assert evaluator.cache_info().opinion.hits == 1


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_default_binned_mode_matches_the_frozen_experiment_kernel(dtype):
    from experiments.paper.kde_approximation.kernel import BinnedDistanceCalculator

    opinions = np.random.default_rng(13).uniform(-1, 1, 12).astype(dtype)
    record = Record(nx.complete_graph(12, create_using=nx.DiGraph), opinions, epsilon=.35)
    calculator = DistanceCalculator(min_bandwidth=.01, sample_count=12, max_kde_samples=37)
    reference = BinnedDistanceCalculator(calculator)
    prepared = reference.prepare(record.graph, record.opinions[2], .35, 2)
    expected = reference.evaluate(prepared, 4081)
    evaluator = MeasurementEvaluator(calculator=calculator)
    actual = evaluator.point(record, 2)
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=1e-12, atol=1e-14)


def test_missing_edges_remain_nonfinite_and_are_not_filled_with_zero():
    record = Record(nx.empty_graph(4, create_using=nx.DiGraph))
    calculator = DistanceCalculator(min_bandwidth=.01, sample_count=4)
    evaluator = MeasurementEvaluator(method="exact", calculator=calculator)
    ip, isub, ih = evaluator.point(record, 0)
    assert np.isfinite(ip) and np.isnan(isub) and np.isnan(ih)


@pytest.mark.parametrize("step", [-1, .5, True])
def test_invalid_steps_do_not_silently_wrap_the_saved_history(step):
    evaluator = MeasurementEvaluator(calculator=Calculator())
    with pytest.raises(ValueError, match="nonnegative integer"):
        evaluator.point(Record(), step)


def test_explicit_sampler_uses_point_without_copying_function_globals(monkeypatch):
    from ehk.metrics import trajectory

    calls = []

    def sample(end, point):
        calls.append(end)
        values = np.array([point(0), point(end)])
        return np.array([0, end]), values[:, 2], values[:, 0], values[:, 1]

    monkeypatch.setattr(trajectory, "sample_indices", sample)
    record = Record()
    result = MeasurementEvaluator(calculator=Calculator()).indices(record)
    assert calls == [4] and record.graph_calls == 2
    np.testing.assert_array_equal(result[0], [0, 4])
