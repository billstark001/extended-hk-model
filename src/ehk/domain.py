"""Typed EHK experiment designs, observations, provenance and storage contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar
from common.experiments import lifecycle

from smp_bindings.domain import (
    ArmObservation, Endpoint, Exposure, HKParams, Pathway, PersistentJob, PolicyChange,
    RecommenderParams, ResearchFrame, Rng, ScenarioMetadata, SlotCounts, TerminalResult,
    TraceReceipt, TrajectoryPrecision, WireRecord, parse_dataclass, parse_metadata,
    parse_observation, to_wire,
)


@dataclass(kw_only=True, slots=True)
class Coordinates(WireRecord):
    case: str | None = None
    replicate: int | None = None
    rewiring: float | None = None
    influence: float | None = None
    repost: float | None = None
    recommendation: str | None = None
    tolerance: float | None = None


@dataclass(kw_only=True, slots=True)
class Task(WireRecord):
    id: str
    metadata: ScenarioMetadata
    setting: str | None = None
    sensitivity: str | None = None
    replicate: int | None = None
    coordinates: Coordinates | None = None


@dataclass(kw_only=True, slots=True)
class LateActivity(WireRecord):
    opinion_shift: float
    rewires: float
    earliest_step: int
    intervals: int


@dataclass(kw_only=True, slots=True)
class Sensitivity(WireRecord):
    axis: str
    epsilon: float
    history: int
    setting: tuple[float, float]
    repost: float


@dataclass(kw_only=True, slots=True)
class RecordingProtocol(WireRecord):
    backend: str
    AgentNumber: bool
    OpinionSum: bool
    RewiringEvent: bool
    PostEvent: bool
    ViewPostsEvent: bool
    ViewPostsDetails: bool
    TrajectoryPrecision: TrajectoryPrecision
    prefix: str
    late_trace: str | None = field(default=None, metadata={"emit_none": True})


@dataclass(kw_only=True, slots=True)
class CampaignProtocol(WireRecord):
    order: tuple[str, ...]
    max_native_workers: int
    timing_requires: str


@dataclass(kw_only=True, slots=True)
class PrimaryOutcome(WireRecord):
    metric: str
    direction: str
    comparison: str


@dataclass(kw_only=True, slots=True)
class ArmConfiguration(WireRecord):
    recommender: str
    parameters: RecommenderParams


@dataclass(kw_only=True, slots=True)
class RecommendationProtocol(WireRecord):
    replicates: int
    settings: tuple[tuple[float, float], ...]
    repost_rates: tuple[float, ...]
    core_epsilon: float
    core_history: int
    sensitivities: tuple[Sensitivity, ...]
    cadence: int
    recording: RecordingProtocol
    campaign: CampaignProtocol
    exposure: str
    separation_threshold: float
    excess_threshold: float
    cross_camp_margin_in_epsilon: float
    reference_horizon: int
    terminal_major_mass: float
    reference: str | None = None
    switch: str | None = None
    window: int | None = None
    checkpoints: tuple[int, ...] | None = None
    late_activity: LateActivity | None = None
    min_stage_effect: float | None = None
    fixed_trigger: int | None = None
    latest_trigger: int | None = None
    primary_outcome: PrimaryOutcome | None = None
    gate_evidence: str | None = None
    variants: dict[str, tuple[str, ...]] | None = None
    variant_arms: dict[str, ArmConfiguration] | None = None


@dataclass(kw_only=True, slots=True)
class ControlProtocol(WireRecord):
    cases: tuple[str, ...]
    replicates: int
    horizon: int
    snapshot_steps: tuple[int, ...]
    arms: tuple[str, ...]
    seed_namespace: str
    pairing: str
    classification: str | None = field(default=None, metadata={"emit_none": True})


@dataclass(kw_only=True, slots=True)
class OriginalProtocol(WireRecord):
    seed_namespace: str
    rng: str
    axes: dict[str, tuple[str | int | float, ...]]
    legacy_unique_names: bool


P = TypeVar("P")


@dataclass(kw_only=True, slots=True)
class Plan(WireRecord, Generic[P]):
    schema: int
    study: str
    profile: str
    protocol: P
    tasks: tuple[Task, ...]
    plan_fingerprint: str | None = None


def parse_task(wire):
    return parse_dataclass(Task, wire)


def parse_plan(wire):
    from common.io.json import fingerprint

    body = dict(wire)
    recorded = body.pop("plan_fingerprint", None)
    if recorded is not None and recorded != fingerprint(body):
        raise ValueError("frozen plan fingerprint differs from its content")
    profile = body["profile"]
    if profile in ("staged", "staged_timing", "recommender_variants"):
        protocol = parse_dataclass(RecommendationProtocol, body["protocol"])
    elif profile == "paired_process_controls":
        protocol = parse_dataclass(ControlProtocol, body["protocol"])
    elif profile in ("epsilon", "gradation", "mech", "replicate"):
        protocol = parse_dataclass(OriginalProtocol, body["protocol"])
    else:
        raise ValueError(f"unsupported EHK study profile: {profile}")
    unknown = body.keys() - {"schema", "study", "profile", "protocol", "tasks"}
    if unknown:
        raise ValueError(f"unknown plan fields: {sorted(unknown)}")
    tasks = tuple(parse_task(item) for item in body["tasks"])
    if len({item.id for item in tasks}) != len(tasks):
        raise ValueError("frozen plan contains duplicate task IDs")
    return Plan(schema=body["schema"], study=body["study"], profile=profile,
                protocol=protocol, tasks=tasks, plan_fingerprint=recorded,
                _wire_fields=frozenset(wire))


def read_plan(path):
    import json

    return parse_plan(json.loads(Path(path).read_text()))


def freeze_plan(path, plan: Plan):
    from common.experiments.plans import freeze_plan as freeze_wire_plan

    wire = to_wire(plan)
    wire.pop("plan_fingerprint", None)
    return parse_plan(freeze_wire_plan(Path(path), wire))


@dataclass(kw_only=True, slots=True)
class StageResult(WireRecord):
    checkpoint: Endpoint
    arms: dict[str, ArmObservation]


@dataclass(kw_only=True, slots=True)
class StagedResult(WireRecord):
    reference: ArmObservation
    stages: dict[str, StageResult]


@dataclass(kw_only=True, slots=True)
class FullResult(WireRecord):
    arms: dict[str, ArmObservation]


StudyResult = StagedResult | FullResult


def parse_result(wire):
    return parse_dataclass(StagedResult if "reference" in wire else FullResult, wire)


@dataclass(kw_only=True, slots=True)
class MeanInterval(WireRecord):
    n: int
    mean: float | None
    lo: float | None
    hi: float | None


@dataclass(kw_only=True, slots=True)
class StageGate(MeanInterval):
    planned_core: int
    completed_core: int
    complete: bool
    passed: bool
    minimum_effect: float
    status: str | None = None


@dataclass(kw_only=True, slots=True)
class StageEvidence(WireRecord):
    plan: str
    contrast: StageGate


@dataclass(kw_only=True, slots=True)
class GitProvenance(WireRecord):
    revision: str
    working_tree_diff_sha256: str


@dataclass(kw_only=True, slots=True)
class Binding(WireRecord):
    binary_sha256: str
    stage_evidence: StageEvidence | None = field(default=None, metadata={"emit_none": True})
    arm_binary_sha256: dict[str, str] | None = None
    binary_build_info: str | None = None
    python_version: str | None = None
    python_source_sha256: dict[str, str] | None = None
    python_git: GitProvenance | None = None


def parse_binding(wire):
    return parse_dataclass(Binding, wire)


@dataclass(kw_only=True, slots=True)
class RawAvailability(WireRecord):
    status: str
    missing: tuple[str, ...]


@dataclass(kw_only=True, slots=True)
class ImportProvenance(WireRecord):
    raw_root: str
    import_id: str | None = None
    path: str | None = None
    sha256: str | None = None


@dataclass(kw_only=True, slots=True)
class FileIdentity(WireRecord):
    path: str
    size: int
    mtime_ns: int
    sha256: str | None = None


@dataclass(kw_only=True, slots=True)
class InputIdentity(WireRecord):
    directory: str
    files: tuple[FileIdentity, ...]
    fingerprint: str
    identity_method: str


@dataclass(kw_only=True, slots=True)
class StoredTask(WireRecord):
    schema: int
    task_id: str
    task_fingerprint: str
    plan_fingerprint: str
    result: StudyResult
    raw: dict[str, RawAvailability]
    raw_complete: bool
    source: ImportProvenance | None


@dataclass(kw_only=True, slots=True)
class BundleManifest(WireRecord):
    schema: int
    exported_at_utc: str
    source_dataset: str
    observation_dataset: str
    plan_fingerprint: str
    profile: str
    available_records: int
    completed: int
    planned: int
    files: dict[str, str]


@dataclass(kw_only=True, slots=True)
class AnalysisStratum:
    sensitivity: str
    epsilon: float
    history: int
    setting: str
    repost: float | str


@dataclass(kw_only=True, slots=True)
class CoverageRow(WireRecord):
    setting: str
    sensitivity: str
    epsilon: float
    history: int
    influence: float
    rewiring: float
    repost: float
    planned: int
    completed: int
    stages: dict[str, int] | None = None


@dataclass(kw_only=True, slots=True)
class PooledCoverage:
    rule: str
    tasks: int
    cells: int


@dataclass(kw_only=True, slots=True)
class AnalysisReport(WireRecord):
    study: str
    created_at_utc: str
    source_directory: str
    source_plan: str
    binding: Binding | None
    record_sha256: dict[str, str]
    raw_root: str
    planned: int
    available_records: int
    completed: int
    complete: bool
    missing: tuple[str, ...]
    settings: tuple[CoverageRow, ...]
    pooled_figures: PooledCoverage
    stage_gate: StageGate | None = None
    latest_fallback_count: int | None = None
    primary_outcome: TimingSummary | None = None
    arms: tuple[ArmTotals, ...] | None = None


@dataclass(kw_only=True, slots=True)
class TimingSummary(MeanInterval):
    metric: str
    contrast: str


@dataclass(kw_only=True, slots=True)
class ArmTotals(WireRecord):
    arm: str
    planned: int
    completed: int


@dataclass(kw_only=True, slots=True)
class DeferredRun(WireRecord):
    state: str
    reason: str


@dataclass(kw_only=True, slots=True)
class CampaignStatus(WireRecord):
    state: str
    results: dict[str, lifecycle.RunStatus | DeferredRun]
    action: str | None = None
    study: str | None = None
    error: str | None = None


def rows_frame(rows, kind=None):
    """Adapt typed rows to a columnar table without intermediate record dictionaries."""
    import pandas as pd
    from dataclasses import fields

    rows = list(rows)
    if not rows and kind is None:
        return pd.DataFrame()
    definitions = [item for item in fields(kind or type(rows[0])) if not item.name.startswith("_")]
    return pd.DataFrame.from_records(
        [tuple(getattr(row, item.name) for item in definitions) for row in rows],
        columns=[item.name for item in definitions],
    )
