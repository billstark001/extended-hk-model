"""Production defaults and native history wiring retain explicit measurement identity."""

from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np
import pytest

from ehk.metrics.evaluation import MeasurementEvaluator
from ehk.metrics.homophily import normalize_homophily, uniform_concordance_probability
from ehk.metrics.state import MeasurementRecord, MeasurementState
from ehk.domain import HKParams, ScenarioMetadata
from ehk.micro import observables
from smp_bindings.domain import Pathway, ProbeResponse, ProbeState, Rng


@dataclass
class SavedRecord:
    metadata: ScenarioMetadata
    max_step: int
    opinions: np.ndarray | None = None
    get_graph: Callable | None = None
    evaluate_probe: Callable | None = None
    directory: str | None = None


def _metadata():
    return ScenarioMetadata(UniqueName="unit", HKParams=HKParams(
        Tolerance=.45, Influence=.1, RewiringRate=.1, RepostRate=0.))


def _probe(end, states):
    return ProbeResponse(version=1, rng=Rng(Algorithm="pcg", Seed1=1, Seed2=2),
                         dynamics_type="hk", source_max_step=end,
                         results=tuple(ProbeState(step=step, precision="float64", points=None,
                             pathway=Pathway(separation=separation, pair_concordance=.5,
                                             edge_concordance=.5 + excess, excess_concordance=excess))
                                       for step, separation, excess in states))


def test_homophily_does_not_assign_an_index_when_the_reference_is_degenerate():
    assert uniform_concordance_probability(2) == 1
    for rho in (0., .5, 1.):
        with pytest.raises(ValueError, match="baseline strictly below one"):
            normalize_homophily(rho, 2)
    assert normalize_homophily(.399375, .45) == pytest.approx(0.)
    assert normalize_homophily(1., .45) == 1.


def test_micro_distance_adopts_4081_and_retains_explicit_exact():
    graph = nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (3, 0)])
    opinions = np.tile(np.array([-.71, -.13, .27, .83]), (14, 1))
    record = SavedRecord(opinions=opinions, metadata=_metadata(),
                             get_graph=lambda step: graph, max_step=13)
    expected = observables.distance_collector.calculate_indices(
        MeasurementState.from_graph(graph, opinions[13]),
        .45, random_state=13, method="binned", bins=4081,
    )
    assert observables.calc_distance(record, 13) == expected
    exact = MeasurementEvaluator(method="exact", calculator=observables.distance_collector)
    measured = MeasurementRecord.from_saved_record(record)
    assert exact.distance(measured, 13) == observables.distance_collector.calculate_indices(
        MeasurementState.from_graph(graph, opinions[13]), .45, random_state=13, method="exact",
    )
    assert observables.calc_distance(record, 13, evaluator=exact) == exact.distance(measured, 13)
    assert observables.distance_collector.density_method == "binned"
    assert max(abs(a - b) for a, b in zip(expected, exact.distance(measured, 13))) > 1e-8


def test_pathway_uses_one_native_run_request_without_python_graphs():
    calls = []

    def measured(steps, **parameters):
        calls.append((steps, parameters))
        return _probe(32, ((0, 0., 0.), (16, .5, 0.), (32, .5, .2)))

    record = SavedRecord(max_step=32, metadata=_metadata(), evaluate_probe=measured)
    assert observables.get_pathway_passage(record, "native") == ("PbS", 16, 32)
    assert calls == [([0, 16, 32], {"measurements": ["pathway"], "epsilon": .45, "binary_path": "native"})]


def test_native_pathway_cannot_silently_use_another_history():
    record = SavedRecord(max_step=32, metadata=_metadata(),
                         evaluate_probe=lambda *args, **kwargs: _probe(31, ()))
    with pytest.raises(ValueError, match="requested saved history"):
        observables.get_pathway_passage(record, "native")
