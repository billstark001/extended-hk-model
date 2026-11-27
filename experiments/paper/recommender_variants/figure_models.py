"""Typed portable observations and layout for the factorial review figures."""

from dataclasses import dataclass, field

from ehk.domain import MeanInterval
from .analysis import FactorialSummary


@dataclass(kw_only=True)
class FactorialObservation:
    task_id: str
    arm: str
    setting: str
    influence: float
    rewiring: float
    repost: float
    end_separation: float
    end_excess: float
    pathway: str


@dataclass(kw_only=True)
class ReviewStratum:
    name: str
    title: str
    observations: tuple[FactorialObservation, ...]
    factorial: tuple[FactorialSummary, ...]


@dataclass(kw_only=True)
class ReviewPlotData:
    schema: int
    plan_fingerprint: str
    bundle_manifest_sha256: str
    source_completed: int
    source_planned: int
    arms: tuple[str, ...]
    labels: tuple[str, ...]
    strata: tuple[ReviewStratum, ...]
    cohort: str


@dataclass(kw_only=True)
class ReviewLayout:
    schema: int = 1
    font_size: float = 9
    summary_size: tuple[float, float] = (8.6, 6.8)
    cell_size: tuple[float, float] = (10.4, 7.3)
    cell_limit: float | None = field(default=None, metadata={"emit_none": True})
    cmap: str = "RdBu_r"
    strata: tuple[str, ...] | None = None
    cell_annotation_size: float | None = None


@dataclass(frozen=True)
class MatchedCoverage:
    matched: int
    planned: int


@dataclass(kw_only=True)
class MatchedMean(MeanInterval):
    sensitivity: str
    epsilon: float
    history: int
    setting: str
    repost: float | str
    arm: str
    metric: str


@dataclass(kw_only=True)
class MatchedContrast(MatchedMean):
    control: str
