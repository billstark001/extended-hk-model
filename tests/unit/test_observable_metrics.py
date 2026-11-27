"""Production defaults and native history wiring retain explicit measurement identity."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from ehk.metrics.evaluation import MeasurementEvaluator
from ehk.metrics.state import MeasurementState
from ehk.micro import observables


def test_micro_distance_adopts_4081_and_retains_explicit_exact():
    graph = nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (3, 0)])
    opinions = np.tile(np.array([-.71, -.13, .27, .83]), (14, 1))
    record = SimpleNamespace(opinions=opinions, metadata={"HKParams": {"Tolerance": .45}},
                             get_graph=lambda step: graph, max_step=13)
    expected = observables.distance_collector.calculate_indices(
        MeasurementState.from_graph(graph, opinions[13]),
        .45, random_state=13, method="binned", bins=4081,
    )
    assert observables.calc_distance(record, 13) == expected
    exact = MeasurementEvaluator(method="exact", calculator=observables.distance_collector)
    assert exact.distance(record, 13) == observables.distance_collector.calculate_indices(
        MeasurementState.from_graph(graph, opinions[13]), .45, random_state=13, method="exact")
    assert observables.distance_collector.density_method == "binned"
    assert max(abs(a - b) for a, b in zip(expected, exact.distance(record, 13))) > 1e-8


def test_pathway_uses_one_native_run_request_without_python_graphs():
    calls = []

    def measured(steps, **parameters):
        calls.append((steps, parameters))
        return {"source_max_step": 32, "results": [
            {"step": 0, "pathway": {"separation": 0., "excess_concordance": 0.}},
            {"step": 16, "pathway": {"separation": .5, "excess_concordance": 0.}},
            {"step": 32, "pathway": {"separation": .5, "excess_concordance": .2}},
        ]}

    record = SimpleNamespace(max_step=32, metadata={"HKParams": {"Tolerance": .45}}, evaluate_probe=measured)
    assert observables.get_pathway_passage(record, "native") == ("PbS", 16, 32)
    assert calls == [([0, 16, 32], {"measurements": ["pathway"], "epsilon": .45, "binary_path": "native"})]


def test_native_pathway_cannot_silently_use_another_history():
    record = SimpleNamespace(max_step=32, metadata={"HKParams": {"Tolerance": .45}},
                             evaluate_probe=lambda *args, **kwargs: {"source_max_step": 31, "results": []})
    with pytest.raises(ValueError, match="requested saved history"):
        observables.get_pathway_passage(record, "native")
