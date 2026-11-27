"""Typed contracts for the numerical KDE study; dictionaries are homogeneous maps."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from ehk.metrics.protocol import MetricProtocol
from experiments.paper.metric_comparison.contracts import (ArtifactManifest, SourceConfiguration, CollectionReceipt,
    PreparedPath, PreparedPoint, PathKey, ReportManifest, PreparationManifest)
from .kernel import KernelDiagnostics

@dataclass(frozen=True, slots=True)
class AdaptiveConfiguration:
    tolerance: float
    max_interval: int
    full_prefix: bool

@dataclass(frozen=True, slots=True)
class ActivityConfiguration:
    active_threshold: float
    min_inactive_value: float

@dataclass(frozen=True, slots=True)
class NumericalSources:
    sources: dict[str,str]
    canonical_protocol: SourceConfiguration
    reference_method: str
    versions: dict[str,str]
    kernel: str
    seed: str
    adaptive: AdaptiveConfiguration
    activity: ActivityConfiguration
    baseline_activity_audit: str
    timing: str

@dataclass(frozen=True, slots=True)
class ReportIdentity:
    directory: str
    sha256: str

@dataclass(frozen=True, slots=True)
class BaselineEvidence:
    endpoint_complement: bool
    preparation_sha256: str
    run_status_sha256: str
    collection_sha256: str
    collection_configuration_fingerprint: str
    comparison_sha256: str
    baseline_report: ReportIdentity

@dataclass(frozen=True, slots=True)
class Selection:
    mode: str
    max_tasks: int | None
    available_tasks: int
    selected_tasks: int
    available_paths: int
    selected_paths: int
    task_keys: tuple[tuple[str,str],...]
    profile_tasks: dict[str,int]
    path_keys: tuple[PathKey,...]
    covered_features: tuple[tuple[str,str],...]
    uncovered_features: tuple[tuple[str,str],...]
    selection_rule: str
    adaptive_stress_rule: str

@dataclass(frozen=True, slots=True)
class BenchmarkKey(PathKey):
    step: int

@dataclass(frozen=True, slots=True, kw_only=True)
class Configuration:
    schema: int
    preparation: str
    baseline: str
    endpoint_complement: bool
    preparation_sha256: str
    run_status_sha256: str
    collection_sha256: str
    collection_configuration_fingerprint: str
    comparison_sha256: str
    baseline_report: ReportIdentity
    selection: Selection
    selected_paths: tuple[PathKey,...]
    bins: tuple[int,...]
    benchmark_states: tuple[BenchmarkKey,...]
    repeats: int
    metrics: NumericalSources
    metric_protocol: MetricProtocol
    exact_reference: str

@dataclass(frozen=True, slots=True)
class DiagnosticObservation:
    diagnostic: KernelDiagnostics
    same_input_sha: str

@dataclass(frozen=True, slots=True)
class PointRow(PathKey):
    bins: int
    step: int
    grid: str
    exact_Ip: float | None
    exact_Is: float | None
    exact_Ih: float | None
    source_Ip: float | None
    source_Is: float | None
    source_Ih: float | None
    approx_Ip: float
    approx_Is: float
    approx_Ih: float
    diagnostics: KernelDiagnostics
    same_input_sha: str

@dataclass(frozen=True, slots=True)
class PathRow(PathKey):
    bins: int
    start_step: int
    end_step: int
    endpoint_complement: bool
    baseline_Iw: float
    source_Iw: float
    fixed_Iw: float
    adaptive_Iw: float
    baseline_steps_json: str
    candidate_steps_json: str
    baseline_active_step: int
    fixed_active_step: int
    adaptive_active_step: int
    baseline_active_threshold: float
    fixed_active_threshold: float
    adaptive_active_threshold: float
    activity_audit_status: str
    extra_exact_calls: int
    extra_exact_seconds: float
    shared_prepare_seconds: float
    evaluate_seconds: float
    candidate_path_seconds: float

@dataclass(frozen=True, slots=True)
class TimingRow(PathKey):
    case_id: str
    step: int
    bins: int
    repeat: int
    same_input_sha: str
    timing_order: str
    exact_prepare_seconds: float
    exact_evaluate_seconds: float
    exact_total_seconds: float
    prepare_seconds: float
    evaluate_seconds: float
    binned_total_seconds: float
    exact_Ip: float
    exact_Is: float
    approx_Ip: float
    approx_Is: float
    diagnostics: KernelDiagnostics

@dataclass(frozen=True, slots=True)
class PathResult:
    points: tuple[PointRow,...]
    paths: tuple[PathRow,...]
    benchmarks: tuple[TimingRow,...]

@dataclass(frozen=True, slots=True)
class Receipt:
    configuration_fingerprint: str
    identity: PathKey
    raw_sha256: dict[str,str]
    baseline_receipt_sha256: str
    result: PathResult

@dataclass(frozen=True, slots=True)
class BaselinePoint(PreparedPoint):
    I_p: float
    I_s: float
    I_h: float

@dataclass(frozen=True, slots=True)
class PathEntry:
    path: PreparedPath
    points: tuple[BaselinePoint,...]
    baseline: CollectionReceipt
    baseline_receipt_sha256: str
    benchmark_steps: tuple[int,...]

@dataclass(frozen=True, slots=True)
class Job:
    id: tuple[str,str]
    bins: tuple[int,...]
    repeats: int
    endpoint_complement: bool
    configuration_fingerprint: str
    entries: tuple[PathEntry,...]
    output: Path

@dataclass(frozen=True, slots=True)
class PathStatus(PathKey):
    status: str
    missing_files: tuple[str,...] = ()
    error: str | None = None

@dataclass(frozen=True, slots=True)
class GroupResult:
    paths: tuple[PathStatus,...]
    receipts: tuple[Receipt,...]

@dataclass(frozen=True, slots=True, kw_only=True)
class CollectionManifest(ArtifactManifest):
    status: str
    planned_paths: int
    completed_paths: int
    endpoint_complement: bool
    pending_paths: int
    failed_paths: int
    selection: Selection
    paths: tuple[PathStatus,...]
    configuration_sha256: str

@dataclass(frozen=True, slots=True)
class Evidence:
    preparation: PreparationManifest
    baseline: ReportManifest
    collection: CollectionManifest
    configuration: Configuration

@dataclass(frozen=True, slots=True, kw_only=True)
class Report(ArtifactManifest):
    status: str
    scope: Selection
    endpoint_complement: bool
    selected_tasks: int
    selected_paths: int
    baseline_paths: int
    baseline_points: int
    bins: tuple[int,...]
    measured_rows: int
    preparation_sha256: str
    baseline_report_sha256: str
    collection_sha256: str
    configuration_sha256: str
    report_source_sha256: str
    paired_analysis_source_sha256: str


@dataclass(frozen=True, slots=True)
class RunPaths:
    preparation: str
    baseline: str
    collection: str
    report: str

@dataclass(frozen=True, slots=True)
class RunStatus:
    schema: int
    state: str
    phase: str
    runner_sha256: str
    endpoint_complement: bool
    paths: RunPaths
    collection: CollectionManifest | None = None
    report: Report | None = None
    error: str | None = None
