"""Explicit records for original-paper analysis and its persisted provenance."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .records import ScenarioStatistics

from ehk.metrics.protocol import MetricProtocol, RecordedMetricProtocol
from ehk.domain import ScenarioMetadata, InputIdentity, WireRecord
from common.experiments.lifecycle import RunStatus


@dataclass(frozen=True, slots=True)
class ProbeIdentity:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FirstPassage:
    cadence: int
    separation: float | None = None
    excess_concordance: float | None = None


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    common_statistics: tuple[str, ...] = ()
    metric_revision: int | None = None
    endpoint_complement: bool = True
    metadata_source: str = "record"
    first_passage: FirstPassage | None = None
    opinion_peak_distance: int | None = None
    active_threshold: float = .98
    min_inactive_value: float = .75
    extra_statistics: str | None = None
    skip_existing: str | None = None


@dataclass(kw_only=True, slots=True)
class AnalysisProtocol(WireRecord):
    metrics: MetricProtocol | RecordedMetricProtocol | None = None
    endpoint_complement: bool | None = None
    common_statistics: tuple[str, ...] = ()
    metric_revision: int | None = None
    metadata_source: str | None = None
    first_passage: FirstPassage | None = None
    opinion_peak_distance: int | None = None
    active_threshold: float | None = None
    min_inactive_value: float | None = None
    extra_statistics: str | None = None
    skip_existing: str | None = None
    analysis_sources: dict[str, str] = field(default_factory=dict)
    binding_sources: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    native_probe: ProbeIdentity | None = None

    @classmethod
    def from_wire(cls, value):
        from ehk.metrics.protocol import parse_metric_protocol

        values = dict(value)
        if values.get("metrics") is not None:
            values["metrics"] = parse_metric_protocol(values["metrics"])
        if values.get("first_passage") is not None:
            values["first_passage"] = FirstPassage(**values["first_passage"])
        if values.get("native_probe") is not None:
            values["native_probe"] = ProbeIdentity(**values["native_probe"])
        if "common_statistics" in values:
            values["common_statistics"] = tuple(values["common_statistics"])
        return cls(**values, _wire_fields=frozenset(value))


@dataclass(kw_only=True, slots=True)
class AnalysisOutcome(RunStatus):
    analysis_id: str
    database: str

    @classmethod
    def from_journal(cls, value: RunStatus, analysis_id, database):
        """Attach analysis identity to the shared journal's completed attempt."""
        return cls(planned=value.planned, completed=value.completed, missing=value.missing,
                   state=value.state, records=value.records,
                   analysis_id=analysis_id, database=str(database))


@dataclass(frozen=True, slots=True)
class AnalysisIdentity:
    mode: str
    protocol: AnalysisProtocol


@dataclass(frozen=True, slots=True)
class AnalysisSelection:
    database: str
    origin: str
    mode: str
    analysis_id: str
    selection: str


@dataclass(frozen=True, slots=True)
class AnalysisTask:
    id: str
    metadata: ScenarioMetadata


@dataclass(frozen=True, slots=True)
class AnalysisJob:
    id: str
    metadata: ScenarioMetadata
    previous: ScenarioStatistics | None
    source: InputIdentity
    task_fingerprint: str


@dataclass(frozen=True, slots=True)
class AnalysisPlan:
    profile: str
    workspace: str
    analysis_id: str
    analysis_protocol: AnalysisProtocol
    tasks: tuple[AnalysisTask, ...]
    schema: int = 3
    study: str = "paper_statistics"


@dataclass(frozen=True, slots=True)
class PlanSummary:
    mode: str
    database: str
    workspace: str
    planned: int
    workspace_available: bool
    metric_revision: int
    include_pathway_probe: bool
    endpoint_complement: bool
    metadata_source: str
    tasks: tuple[ScenarioMetadata, ...]
    state: str = "planned"
