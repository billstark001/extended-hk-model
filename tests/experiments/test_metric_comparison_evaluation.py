"""Exact caching preserves ordered inputs, absolute-step seeds and adaptive paths."""

import copy
import subprocess
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from ehk.metrics.trajectory import sample_indices
from ehk.micro import observables
from experiments.paper.metric_comparison.evaluation import ExactMetricCache


@pytest.fixture(autouse=True)
def no_native_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("metric evaluation must not launch a simulator or a probe")

    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _record(*, dtype=np.float64, directory="raw/one", horizon=4):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from(((0, 1), (0, 2), (1, 2), (2, 3), (3, 0)))
    return SimpleNamespace(
        directory=directory, max_step=horizon,
        opinions=np.tile(np.array([-.8, -.2, .1, .7], dtype=dtype), (horizon + 1, 1)),
        metadata={"HKParams": {"Tolerance": .45}},
        graph=graph, get_graph=lambda step: graph,
    )


def _count_metrics(monkeypatch):
    calls = []

    def distance(record, step):
        calls.append(("distance", record.directory, step))
        return step / 100, float(record.opinions[step, 0])

    def homophily(record, step):
        calls.append(("homophily", record.directory, step))
        return record.metadata["HKParams"]["Tolerance"]

    monkeypatch.setattr(observables, "calc_distance_components", lambda record, step: (*distance(record, step), 0., 0.))
    monkeypatch.setattr(observables, "calc_homophily", homophily)
    return calls


def test_equal_states_share_joint_scalar_results_across_records(monkeypatch):
    cache = ExactMetricCache()
    calls = _count_metrics(monkeypatch)
    first, branch = _record(), _record(directory="raw/branch")
    assert cache.distance(first, 2) == (.02, -.8)
    assert cache.homophily(first, 2) == .45
    assert cache.distance(branch, 2) == (.02, -.8)
    assert cache.homophily(branch, 2) == .45
    assert calls == [("distance", "raw/one", 2), ("homophily", "raw/one", 2)]
    assert (len(cache), cache.hits, cache.misses) == (1, 3, 1)


@pytest.mark.parametrize("change", ["step", "epsilon", "opinions", "dtype", "opinion_order", "edge_order", "edges"])
def test_distinct_exact_inputs_never_reuse_values(monkeypatch, change):
    calls = _count_metrics(monkeypatch)
    cache = ExactMetricCache()
    first, second = _record(), _record(directory="raw/other")
    step = 1
    cache.distance(first, step)
    if change == "step":
        step = 2
    elif change == "epsilon":
        second.metadata["HKParams"]["Tolerance"] = .35
    elif change == "opinions":
        second.opinions[step, 0] = np.nextafter(second.opinions[step, 0], 0)
    elif change == "dtype":
        second.opinions = second.opinions.astype(np.float32)
    elif change == "opinion_order":
        second.opinions[step] = second.opinions[step, ::-1]
    elif change == "edge_order":
        second.graph.remove_edge(0, 1)
        second.graph.add_edge(0, 1)
        assert set(first.graph.edges) == set(second.graph.edges)
        assert list(first.graph.edges) != list(second.graph.edges)
    else:
        second.graph.add_edge(1, 3)
    cache.distance(second, step)
    assert len(calls) == 4
    assert (len(cache), cache.misses) == (2, 2)


def test_lru_capacity_is_bounded_and_evicts_only_the_least_recent_key(monkeypatch):
    _count_metrics(monkeypatch)
    cache = ExactMetricCache(max_entries=2)
    record = _record()
    cache.distance(record, 0)
    cache.distance(record, 1)
    cache.distance(record, 0)
    cache.distance(record, 2)
    assert len(cache) == 2
    cache.distance(record, 0)
    assert cache.misses == 3
    cache.distance(record, 1)
    assert len(cache) == 2 and cache.misses == 4


def test_default_capacity_retains_at_most_512_joint_results(monkeypatch):
    _count_metrics(monkeypatch)
    cache = ExactMetricCache()
    record = _record(horizon=512)
    for step in range(513):
        cache.distance(record, step)
    assert len(cache) == 512 and cache.misses == 513
    cache.homophily(record, 512)
    assert cache.misses == 513
    cache.homophily(record, 0)
    assert len(cache) == 512 and cache.misses == 514


def test_actual_distance_entry_keeps_absolute_step_as_its_sampling_seed(monkeypatch):
    seeds = []

    def calculate(graph, opinions, epsilon, *, random_state):
        seeds.append(random_state)
        return SimpleNamespace(rand_o=random_state / 100, rand_s=.2, worst_o=.3, worst_s=.4)

    monkeypatch.setattr(observables.distance_collector, "calculate", calculate)
    cache = ExactMetricCache()
    record = _record()
    assert cache.distance(record, 1) == (.01, .2)
    assert cache.distance(record, 2) == (.02, .2)
    assert cache.distance(record, 1) == (.01, .2)
    assert seeds == [1, 2]


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        ExactMetricCache(max_entries=capacity)


def test_path_local_mode_supports_instrumented_records_without_state_access(monkeypatch):
    calls = []
    cache = ExactMetricCache(cross_records=False)
    monkeypatch.setattr(observables, "calc_distance_components", lambda record, step: (calls.append(step) or .1, .2, .3, .4))
    monkeypatch.setattr(observables, "calc_homophily", lambda record, step: .3)
    first = SimpleNamespace(directory="raw/first")
    second = SimpleNamespace(directory="raw/second")
    assert cache.distance(first, 4) == (.1, .2)
    assert cache.homophily(first, 4) == .3
    assert cache.distance(second, 4) == (.1, .2)
    assert calls == [4, 4]


def test_indices_use_explicit_measurements_without_patching_global_functions(monkeypatch):
    cache = ExactMetricCache(cross_records=False)
    calls = _count_metrics(monkeypatch)
    record = _record()
    def original(record):
        pytest.fail("exact sampling must not dispatch through the production binned selector")

    monkeypatch.setattr(observables, "get_indices", original)
    distance, homophily = observables.calc_distance_components, observables.calc_homophily
    first, second = cache.indices(record), cache.indices(record)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first[0], [0, 4])
    np.testing.assert_array_equal(first[2], [0, .04])
    assert len(calls) == 6
    assert observables.calc_distance_components is distance and observables.calc_homophily is homophily
    assert observables.get_indices is original


def test_cached_adaptive_steps_values_and_signed_integral_equal_the_existing_algorithm():
    record = _record(horizon=32)
    for step in range(33):
        record.opinions[step] = record.opinions[step] * (1 - .012 * step)
    original = sample_indices(record.max_step, lambda step: (
        *observables.calc_distance_components(record, step)[:2], observables.calc_homophily(record, step),
    ))
    cache = ExactMetricCache()
    # Prepopulate an exported observation that adaptive evaluation may also query.
    cache.distance(record, 16)
    cached = cache.indices(record)
    repeated = cache.indices(copy.copy(record))
    for expected, actual, again in zip(original, cached, repeated):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(again, expected)
    assert observables.get_gradation_index_hp(cached[2], cached[1]) == observables.get_gradation_index_hp(
        original[2], original[1],
    )
    assert cache.hits > 0
