"""Typed snapshots, measurements and receipts for process controls."""
from __future__ import annotations
from dataclasses import dataclass
from smp_bindings.domain import SavedPost
from ehk.domain import InputIdentity, ScenarioMetadata
from ehk.metrics.protocol import MetricProtocol
from experiments.paper.analysis.models import AnalysisProtocol

@dataclass(frozen=True, slots=True)
class SnapshotPosts:
    sharer: int
    entries: tuple[SavedPost, ...]

@dataclass(frozen=True, slots=True)
class SnapshotInput:
    schema: int
    step: int
    dynamics: str
    node_ids: tuple[int, ...]
    opinions: tuple[float, ...]
    edges: tuple[tuple[int, int], ...]
    posts: tuple[SnapshotPosts, ...]

@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    pair_id: str
    case: str
    replicate: int
    arm: str
    step: int
    checkpoint: str
    checkpoint_sha256: str
    snapshot_key: str
    state_fingerprint: str

@dataclass(frozen=True, slots=True)
class SnapshotIndex:
    schema: int
    plan_fingerprint: str
    snapshots: tuple[SnapshotEntry, ...]

@dataclass(frozen=True, slots=True)
class IndexRow:
    pair_id: str
    case: str
    replicate: int
    arm: str
    step: int
    I_p: float
    I_s: float
    I_h: float | None

@dataclass(frozen=True, slots=True)
class IntegralRow:
    pair_id: str
    case: str
    replicate: int
    arm: str
    horizon: int
    I_w: float
    endpoint_complement: bool
    sampling: str = "paper adaptive grid"

@dataclass(frozen=True, slots=True)
class ArmIdentity:
    plan_fingerprint: str
    pair_id: str
    arm: str
    source: InputIdentity
    analysis: AnalysisProtocol

@dataclass(frozen=True, slots=True)
class ArmJob:
    dataset: str
    metadata: ScenarioMetadata
    case: str
    replicate: int
    horizon: int
    endpoint_complement: bool
    identity: ArmIdentity
    key: str

    @property
    def id(self):
        return f"{self.identity.pair_id}/{self.identity.arm}"

@dataclass(frozen=True, slots=True)
class ArmMeasurement:
    identity: ArmIdentity
    indices: tuple[IndexRow, ...]
    integral: IntegralRow

@dataclass(frozen=True, slots=True)
class MetricReceipt:
    plan_fingerprint: str
    snapshots: int
    metric_implementation_sha256: str
    classification: None
    metrics: tuple[str, ...]
    endpoint_complement: bool
    metric_protocol: MetricProtocol
    analysis: AnalysisProtocol

@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    snapshots: int
    index_rows: int
    arms: int
    state: str = "complete"
