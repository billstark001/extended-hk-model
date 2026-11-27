"""Computed paired-study rows, independent of their table serialization."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from ehk.domain import AnalysisStratum, Exposure, MeanInterval, Pathway


@dataclass(kw_only=True, slots=True)
class ExposureColumns:
    updates: int
    requested: int
    neighbor_seen: int
    neighbor_concordant: int
    neighbor_with_posts: int
    neighbor_share_sum: float
    recommended: int
    cross_camp: int
    cross_confident: int
    cross_rewire: int
    rewires: int
    absolute_shift: float
    exploration_shown: int
    exploration_cross_camp: int
    exploration_cross_confident: int
    exploration_cross_rewire: int
    policy_shown: int
    policy_cross_camp: int
    policy_cross_confident: int
    policy_cross_rewire: int
    bridge_shown: int
    bridge_cross_camp: int
    bridge_cross_confident: int
    bridge_cross_rewire: int


@dataclass(kw_only=True, slots=True)
class ArmSummary(ExposureColumns):
    start_step: int
    end_step: int
    switch_step: int | None
    switch_fallback: bool
    raw_record: str
    exposure_trace: str | None
    start_separation: float
    end_separation: float
    separation_change: float
    start_excess: float | None
    end_excess: float | None
    excess_change: float | None
    end_edge_concordance: float | None
    end_pair_concordance: float
    outcome: str
    opinion_peaks: int
    opinion_shape: str
    pathway: str
    separation_step: int | None
    segregation_step: int | None
    window_pathway: str
    window_separation_step: int | None
    window_segregation_step: int | None
    start_separation_crossed: bool
    start_segregation_crossed: bool
    fill: float | None
    cross_camp_per_update: float | None
    cross_share: float | None
    followee_post_concordance: float | None
    rewires_per_update: float | None
    opinion_shift_per_update: float | None
    non_exploration_shown: int
    non_exploration_cross_camp: int
    non_exploration_cross_confident: int
    non_exploration_cross_rewire: int


@dataclass(kw_only=True, slots=True)
class RunIdentity:
    task_id: str
    setting: str
    sensitivity: str
    epsilon: float
    history: int
    influence: float
    rewiring: float
    replicate: int
    repost: float
    stage: str
    arm: str
    binary_sha256: str | None
    reference_steps: int | None = None


class ArmFactors(Protocol):
    recommender: str
    zeta: float | None
    score_noise: float | None


@dataclass(kw_only=True, slots=True)
class StudyRunRow:
    identity: RunIdentity
    summary: ArmSummary
    factors: ArmFactors | None = None


@dataclass(kw_only=True, slots=True)
class TrajectoryRow:
    identity: RunIdentity
    step: int
    relative_step: int
    pathway: Pathway
    exposure: ExposureColumns


@dataclass(kw_only=True, slots=True)
class ContrastRow(MeanInterval):
    sensitivity: str
    epsilon: float
    history: int
    setting: str
    repost: float | str
    stage: str
    arm: str
    control: str
    metric: str
