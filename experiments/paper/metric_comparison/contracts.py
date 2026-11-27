"""Actual in-memory contracts for preparation, collection and reporting.

The persisted JSON/SQLite documents are parsed at the boundary. Scientific
tables remain DataFrames; scheduling, provenance and recovery use these records.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeVar

import pandas as pd

from ehk.metrics.protocol import MetricProtocol
from ehk.domain import Binding

T = TypeVar("T")


def table_records(frame: pd.DataFrame, model: type[T]) -> tuple[T, ...]:
    """Decode a scientific table into its declared row schema without dicts."""
    columns = [item.name for item in fields(model)]
    return tuple(model(*row) for row in frame[columns].itertuples(index=False, name=None))


@dataclass(frozen=True, slots=True)
class PathKey:
    profile: str
    task_id: str
    stage: str
    arm: str

    @property
    def label(self) -> str:
        return f"{self.profile}/{self.task_id}/{self.stage}/{self.arm}"


@dataclass(frozen=True, slots=True)
class PathProvenance(PathKey):
    plan_fingerprint: str
    binary_sha256: str
    raw_record: str

    @property
    def key(self) -> PathKey:
        return PathKey(self.profile, self.task_id, self.stage, self.arm)


@dataclass(frozen=True, slots=True)
class PreparedPoint(PathProvenance):
    step: int
    relative_step: int
    separation: float
    excess_concordance: float
    edge_concordance: float
    pair_concordance: float
    prefix_reference_raw_record: str


@dataclass(frozen=True, slots=True)
class PreparedPath(PathProvenance):
    source_dataset: str
    raw_directory: str
    start_step: int
    end_step: int
    full_prefix_required: bool
    prefix_reference_raw_record: str
    epsilon: float
    history: int
    influence: float
    rewiring: float
    repost: float
    sensitivity: str
    replicate: int
    seed1: str
    seed2: str
    available: bool
    metadata_json: str
    seed_json: str
    expected_rec_params_json: str
    missing_files: str
    terminal_status: str | None
    major_components: int | None
    terminal_json: str

    @property
    def identity(self) -> PathProvenance:
        return PathProvenance(self.profile, self.task_id, self.stage, self.arm,
                              self.plan_fingerprint, self.binary_sha256, self.raw_record)


@dataclass(frozen=True, slots=True)
class MetricPoint(PathProvenance):
    step: int
    I_p: float
    I_s: float
    I_h: float


@dataclass(frozen=True, slots=True)
class PathMetric(PathProvenance):
    I_w: float
    sampling_method: str
    prefix_start: int
    trajectory_end: int
    endpoint_complement: bool


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    adaptive_steps: tuple[int, ...]
    endpoint_complement: bool


@dataclass(frozen=True, slots=True)
class PathMeasurement:
    points: tuple[MetricPoint, ...]
    path: PathMetric
    evaluation: EvaluationRecord


@dataclass(frozen=True, slots=True)
class CollectionReceipt:
    configuration_fingerprint: str
    identity: PathProvenance
    endpoint_complement: bool
    raw_sha256: dict[str, str]
    result: PathMeasurement


@dataclass(frozen=True, slots=True)
class PathStatus(PathProvenance):
    status: str
    raw_directory: str = ""
    missing_files: tuple[str, ...] = ()
    error: str | None = None

    @classmethod
    def pending(cls, identity: PathProvenance) -> PathStatus:
        return cls(identity.profile, identity.task_id, identity.stage, identity.arm,
                   identity.plan_fingerprint, identity.binary_sha256, identity.raw_record, "pending")


@dataclass(frozen=True, slots=True)
class PathJob:
    path: PreparedPath
    points: tuple[PreparedPoint, ...]


@dataclass(frozen=True, slots=True)
class CollectionJob:
    id: tuple[str, str]
    output: Path
    raw_roots: dict[str, str]
    configuration_fingerprint: str
    endpoint_complement: bool
    paths: tuple[PathJob, ...]


@dataclass(frozen=True, slots=True)
class CollectionOutcome:
    paths: tuple[PathStatus, ...]
    receipts: tuple[CollectionReceipt, ...]
    cache_hits: int
    cache_misses: int


@dataclass(frozen=True, slots=True)
class DistanceParameters:
    min_bandwidth: float
    sample_count: int
    max_kde_samples: int
    t_err: float
    k: float


@dataclass(frozen=True, slots=True)
class SourceConfiguration:
    collector_sha256: str
    source_sha256: dict[str, str]
    versions: dict[str, str]
    distance_parameters: DistanceParameters
    path_sampling: str
    density_evaluation: str
    opinion_point_sampling: str
    homophily_weighting: str
    prefix: str


@dataclass(frozen=True, slots=True)
class CollectionConfiguration:
    schema: int
    preparation_fingerprint: str
    metrics: SourceConfiguration
    raw_roots: dict[str, str]
    endpoint_complement: bool
    metric_protocol: MetricProtocol


@dataclass(frozen=True, slots=True)
class CacheCounts:
    hits: int = 0
    misses: int = 0


@dataclass(frozen=True, slots=True)
class StatusCounts:
    complete: int = 0
    cached: int = 0
    pending: int = 0
    failed: int = 0

    @classmethod
    def count(cls, statuses) -> StatusCounts:
        statuses = tuple(statuses)
        return cls(*(sum(item.status == state for item in statuses)
                     for state in ("complete", "cached", "pending", "failed")))


@dataclass(frozen=True, slots=True)
class CollectionProgress:
    state: str
    workers: int
    planned_paths: int
    selected_paths: int
    completed_paths: int
    counts: StatusCounts
    elapsed_seconds: float
    cache: CacheCounts
    last_task: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactManifest:
    schema: int
    files: dict[str, str]
    sqlite_tables: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectionManifest(ArtifactManifest):
    preparation_fingerprint: str
    endpoint_complement: bool
    preparation_sha256: str
    status: str
    planned_paths: int
    selected_paths: int
    completed_paths: int
    pending_paths: int
    failed_paths: int
    points: int
    paths: tuple[PathStatus, ...]
    cache: CacheCounts


@dataclass(frozen=True, slots=True)
class PreparationSource:
    bundle: str
    manifest_path: str
    manifest_sha256: str
    plan_fingerprint: str
    profile: str
    planned: int
    available_tasks: int
    missing_tasks: tuple[str, ...]
    raw_dataset: str
    binding: Binding


@dataclass(frozen=True, slots=True)
class Weighting:
    I_h: str
    C_edge: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparationManifest(ArtifactManifest):
    status: str
    sources: tuple[PreparationSource, ...]
    points: int
    paths: int
    available_raw_paths: int
    pending_raw_paths: int
    missing_metrics: tuple[str, ...]
    weighting: Weighting
    path_requirement: str


@dataclass(frozen=True, slots=True)
class Coverage:
    requested: int
    matched: int
    pending: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonManifest(ArtifactManifest):
    status: str
    coverage: dict[str, Coverage]
    endpoint_complement: bool
    preparation_sha256: str
    collection_sha256: str


@dataclass(frozen=True, slots=True)
class AlignedEvidence:
    preparation: PreparationManifest
    comparison: ComparisonManifest


@dataclass(frozen=True, slots=True)
class PeakAudit:
    computed_paths: int
    archived_comparisons: int
    archived_differences: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportManifest(ArtifactManifest):
    status: str
    paths: int
    endpoint_complement: bool
    points: int
    paired_interventions: int
    endpoint_peak_count_audit: PeakAudit
    preparation_sha256: str
    comparison_sha256: str
    report_source_sha256: str


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    preparation: str
    output: str
    collection: str
    alignment: str
    report: str
    collection_progress: str


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    status: str
    planned_paths: int
    selected_paths: int
    completed_paths: int
    pending_paths: int
    failed_paths: int
    points: int
    failures: tuple[PathStatus, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineStatus:
    schema: int
    state: str
    phase: str
    attempt: int
    workers: int
    attempts: int
    make_figures: bool
    endpoint_complement: bool
    runner_sha256: str
    paths: PipelinePaths
    collection: CollectionSummary | None = None
    alignment: ComparisonManifest | None = None
    report: ReportManifest | None = None
    reason: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedTaskRow:
    profile: str
    task_id: str
    plan_fingerprint: str
    task_fingerprint: str
    metadata_json: str


@dataclass(frozen=True, slots=True)
class PreparationInput:
    bundle: str
    manifest_sha256: str
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class PreparationInputs:
    schema: int
    sources: tuple[PreparationInput, ...]
    raw_roots: dict[str, str]
    implementation_sha256: str
    reader_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class PreparedTaskCheckpoint:
    input_fingerprint: str
    task_fingerprint: str
    available: bool
    task: PreparedTaskRow
    points: tuple[PreparedPoint, ...]
    paths: tuple[PreparedPath, ...]


@dataclass(frozen=True, slots=True)
class EndpointRequest(PathKey):
    plan_fingerprint: str
    raw_record: str
    end_step: int


@dataclass(frozen=True, slots=True)
class ArchivedPeak:
    task_id: str
    stage: str
    arm: str
    end_step: int
    opinion_peaks: float
    raw_record: str


@dataclass(frozen=True, slots=True)
class EndpointFeature(PathKey):
    end_step: int
    opinion_peaks: int
    saved_opinion_peaks: float
    saved_count_matches: bool | None
    summary_sha256: str
    summary_path: str
    peak_implementation_sha256: str
    saved_count_table: str


@dataclass(frozen=True, slots=True)
class PathwayInput(PathKey):
    sensitivity: str
    epsilon: float
    history: int
    influence: float
    rewiring: float
    repost: float
    start_step: int
    end_step: int
    followup_steps: int
    I_w: float
    terminal_outcome: str


@dataclass(frozen=True, slots=True)
class ClassifiedPath(PathwayInput):
    pathway: str
    separation_step: int | None
    segregation_step: int | None
    prefix_observations: int


@dataclass(frozen=True, slots=True)
class ReportInputs:
    preparation_sha256: str
    comparison_sha256: str
    source_sha256: dict[str, str]
    make_figures: bool
    allow_partial: bool
