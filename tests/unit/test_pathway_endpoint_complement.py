"""The default pathway tendency retains its explicit endpoint contribution."""

import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from experiments.paper.analysis.models import AnalysisOptions

from common.io.json import fingerprint
from ehk.metrics.protocol import METRIC_PROTOCOL, METRIC_REVISION, metric_protocol
from ehk.metrics.trajectory import pathway_index
from ehk.micro.observables import c
from experiments.paper.analysis.run import _build_parser
from experiments.paper.analysis.statistics import analysis_protocol


def test_default_completion_and_explicit_observed_integral_for_consensus_return():
    p, h = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    # The signed return leg gives -0.5; completion at the final concordance adds 1.
    assert pathway_index(p, h) == pytest.approx(0.5)
    assert pathway_index(p, h, endpoint_complement=False) == pytest.approx(-0.5)


def test_completion_uses_actual_endpoint_without_sorting_or_clipping():
    p, h = np.array([0.0, 0.5, 0.1]), np.array([0.0, 0.0, 1.0])
    assert pathway_index(p, h) == pytest.approx(0.7)
    assert pathway_index(p, h, endpoint_complement=False) == pytest.approx(-0.2)
    # A value above the reference yields a signed negative endpoint contribution.
    assert pathway_index([2.0], [0.5]) == pytest.approx(-0.5)
    assert pathway_index([2.0], [0.5], endpoint_complement=False) == 0.0
    assert pathway_index([0.2], [0.5]) == pytest.approx(0.4)


def test_context_selection_clears_cached_pathway_result():
    context = c.clone(state=True)
    context.set_state(p_index=np.array([0.0, 0.2]), h_index=np.array([0.5, 0.5]), endpoint_complement=True)
    assert context.gradation_index_hp == pytest.approx(0.5)
    context.set_state(endpoint_complement=False)
    assert context.gradation_index_hp == pytest.approx(0.1)
    context.set_state(endpoint_complement=True)
    assert context.gradation_index_hp == pytest.approx(0.5)


def test_protocol_and_cache_identity_distinguish_the_selected_definition():
    completed, observed = metric_protocol(), metric_protocol(endpoint_complement=False)
    assert METRIC_REVISION == 6 and completed.endpoint_complement is True
    assert observed.endpoint_complement is False and observed.endpoint_target is None
    with pytest.raises(FrozenInstanceError):
        observed.distance_kde.bins = 1
    assert METRIC_PROTOCOL.distance_kde.bins == 4081
    default = analysis_protocol()
    enabled = analysis_protocol(AnalysisOptions(endpoint_complement=True))
    disabled = analysis_protocol(AnalysisOptions(endpoint_complement=False))
    assert fingerprint(default) == fingerprint(enabled)
    assert fingerprint(enabled) != fingerprint(disabled)
    assert default.metrics.endpoint_complement is True
    assert disabled.metrics.endpoint_complement is False


def test_analysis_cli_defaults_to_completion_and_accepts_both_switches():
    parser = _build_parser()
    assert parser.parse_args(["gradation", "raw"]).endpoint_complement is True
    assert parser.parse_args(["gradation", "raw", "--endpoint-complement"]).endpoint_complement is True
    assert parser.parse_args(["gradation", "raw", "--no-endpoint-complement"]).endpoint_complement is False


@pytest.mark.parametrize("p,h", [([], []), ([0.0], [0.0, 1.0]), ([[0.0]], [[0.0]])])
def test_completion_rejects_undefined_trajectory_shapes(p, h):
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        pathway_index(p, h)


def test_completion_switch_is_boolean():
    with pytest.raises(TypeError, match="boolean"):
        pathway_index([0.0], [0.5], endpoint_complement=1)
    with pytest.raises(TypeError, match="boolean"):
        metric_protocol(endpoint_complement="false")
