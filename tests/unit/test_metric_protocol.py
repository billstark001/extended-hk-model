"""Typed protocol evidence preserves archived omissions without new defaults."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from ehk.metrics.protocol import (
    MetricProtocol, RecordedMetricProtocol, metric_protocol, parse_metric_protocol,
)
from smp_bindings.domain import to_wire


def test_complete_protocol_roundtrips_as_the_formal_typed_contract():
    protocol = metric_protocol(endpoint_complement=False)
    decoded = parse_metric_protocol(asdict(protocol))
    assert isinstance(decoded, MetricProtocol) and decoded == protocol
    assert decoded.distance_kde.bins == 4081
    with pytest.raises(FrozenInstanceError):
        decoded.endpoint_complement = True


def test_partial_archived_protocol_does_not_gain_new_conventions():
    wire = {"revision": 3}
    archived = parse_metric_protocol(wire)
    assert isinstance(archived, RecordedMetricProtocol)
    assert archived.I_p is None and archived.distance_kde is None
    assert archived.__wire__() == wire
    assert to_wire(archived) == wire


def test_partial_nested_protocol_preserves_missing_and_explicit_null():
    wire = {"revision": 4, "endpoint_target": None, "distance_kde": {"evaluation": "exact", "bins": None}}
    archived = parse_metric_protocol(wire)
    assert archived.distance_kde.output_points is None
    assert archived.__wire__() == wire
    assert to_wire(archived) == wire


@pytest.mark.parametrize("wire", [{}, {"revision": True}, {"revision": 6, "unknown": 1}])
def test_protocol_decoder_rejects_unidentified_or_unknown_evidence(wire):
    with pytest.raises(ValueError):
        parse_metric_protocol(wire)
