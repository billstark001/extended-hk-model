"""Owned lifted contracts. Times use tau; runtime steps are nonnegative integers."""

from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict

from theory.types import ParametersData

Kind = Literal[
    "initial",
    "micro_atlas",
    "micro_tail",
    "micro_probe",
    "micro_moments",
    "meso_atlas",
    "meso_probe",
    "meso_moments",
    "measure",
    "micro_extend",
    "meso_extend",
]
Variant = Literal["stochastic", "deterministic", "rewire_only", "transport_only"]


class Task(TypedDict):
    id: str
    kind: Kind
    parameters: ParametersData
    cohort: Literal["population", "mechanism", "pilot", "smoke"]
    initial: int
    mother: int
    seed: int
    replicates: list[int]
    horizon_steps: int
    observe_steps: list[int]
    source_task: str | None
    source_step: int
    variant: Variant
    layer: Literal["naive", "base"]
    initial_law: Literal["uniform", "bridge"]
    bridge_agents: int
    group_center: float


class Probability(TypedDict):
    paths: int
    counts: list[int]
    unresolved: int
    pending_paths: int
    estimate: list[float]
    lower: list[float]
    upper: list[float]
    disagreement_lower: float
    disagreement_upper: float


class StateReference(TypedDict):
    step: int
    physical_hash: str
    checkpoint: str
    checkpoint_sha256: str
    category: str
    k_all: int


class PathResult(TypedDict):
    replicate: int
    step: int
    category: str
    k_all: int
    stop_reason: str
    first_hit_step: NotRequired[int]
    checkpoint: NotRequired[str]
    checkpoint_sha256: NotRequired[str]
    delta: NotRequired[Any]
    final: NotRequired[dict[str, Any]]


class RuntimeResult(TypedDict):
    states: list[StateReference]
    paths: list[PathResult]
    executed_steps: int
    elapsed_seconds: float
    source_physical_hash: str
    files: dict[str, str]
    binding: dict[str, Any]
    dependency_sha256: str | None
    wall_seconds: float
