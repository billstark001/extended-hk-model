"""Central physical/task/result contracts for theory libraries and their adapters.

Time in frames is physical update time; horizon/record_tau in ParametersData
use tau=alpha*t. Array annotations express dtype, not shape or conservation:
existing runtime validators remain responsible for those scientific conditions.
"""

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

from common.types import Record, RuntimeBinding

Dynamics: TypeAlias = Literal["hk", "deffuant"]
Recommender: TypeAlias = Literal["random", "opinion_random", "structure_random_l0"]
Selection: TypeAlias = Literal["weighted", "ranked"]
TaskKind: TypeAlias = Literal["measure", "micro"]
FloatArray: TypeAlias = NDArray[np.float64]
FloatVector: TypeAlias = FloatArray | Sequence[float]
NumericArray: TypeAlias = FloatVector | Sequence[Sequence[float | None]]


class ParametersData(TypedDict):
    dynamics: Dynamics
    recommender: Recommender
    epsilon: float
    alpha: float
    ratio: float
    degree: int
    recommendations: int
    bins: int
    population: int
    dt: float
    horizon: float
    record_tau: float
    steepness: float
    opinion_tolerance: float
    quadrature: int


class StoppingOptions(TypedDict, total=False):
    mode: Literal["fixed_steps", "state_only", "energy_only", "state_or_energy", "state_and_energy"]
    minimum_steps: int
    check_every: int
    patience_steps: int
    state_l1_tolerance: float
    energy_absolute_tolerance: float
    energy_relative_tolerance: float


class MeasureOptions(TypedDict, total=False):
    initial_mass: FloatVector
    final_state: bool
    snapshot_taus: Sequence[float]
    stopping: StoppingOptions


class MicroTaskFields(TypedDict, total=False):
    initial: int
    branch: int
    initial_seed: int
    process_seed: int
    selection: Selection


class TaskSpecification(MicroTaskFields):
    """Pre-ID task fields; hashing excludes the later id and roles fields."""

    kind: TaskKind
    parameters: ParametersData
    request_options: MeasureOptions


class _TaskIdentity(TypedDict):
    id: str
    parameters: ParametersData
    request_options: MeasureOptions
    roles: list[str]


class MeasureTask(_TaskIdentity):
    kind: Literal["measure"]


class MicroTask(_TaskIdentity):
    kind: Literal["micro"]
    initial: int
    branch: int
    initial_seed: int
    process_seed: int
    selection: Selection


Task: TypeAlias = MeasureTask | MicroTask


class PairIdentity(TypedDict):
    parameters: ParametersData
    request_options: MeasureOptions
    initial: int
    initial_seed: int
    selection: Selection
    binding: RuntimeBinding


class PairTask(PairIdentity):
    id: str


class PairReference(TypedDict):
    key: str
    sha256: str


class ConservationDiagnostics(TypedDict):
    max_node_mass_residual: float
    max_fixed_degree_residual: float
    backend: NotRequired[str]
    state_dimension: NotRequired[int]
    recorded_points: NotRequired[int]
    elapsed_seconds: NotRequired[float]
    opinion_method: NotRequired[str]
    recommender: NotRequired[str]
    executed_steps: NotRequired[int]
    stop_reason: NotRequired[str]
    converged: NotRequired[bool]
    final_state_l1_rate: NotRequired[float]
    final_node_energy_rate: NotRequired[float]
    final_edge_energy_rate: NotRequired[float]
    stable_steps: NotRequired[int]


class _StateFields(TypedDict, total=False):
    rho: FloatVector
    edge: FloatVector
    velocity: FloatVector
    node_potential: FloatVector
    edge_potential: FloatArray | Sequence[float | None]
    reference_energy: float
    variance: float
    discordant: float
    budget_rewire: float
    budget_motion: float


class PairFrame(_StateFields):
    step: int
    time: float
    node_energy: float
    edge_energy: float


class MicroFrame(_StateFields):
    """Exact energies exist at each sample; binned fields only at projections."""

    step: int
    time: float
    exact_node_energy: float
    exact_edge_energy: float
    node_energy: NotRequired[float]
    edge_energy: NotRequired[float]
    rewires: NotRequired[int]
    mean_degree: NotRequired[float]
    exact_variance: NotRequired[float]
    exact_discordant: NotRequired[float]
    projected: NotRequired[bool]


class PairTrace(TypedDict):
    frames: list[PairFrame]
    diagnostics: ConservationDiagnostics
    axis: NotRequired[FloatVector]
    pathway: NotRequired[float | None]
    polarization: NotRequired[FloatVector]
    homophily: NotRequired[FloatVector]


class _MicroIdentity(TypedDict):
    initial_hash: str
    projection_l1: float


class SharedPairResult(_MicroIdentity):
    pair: PairTrace


class MicroResult(_MicroIdentity):
    """Stored branch: legacy embedded pair or a native shared reference."""

    frames: list[MicroFrame]
    pair: NotRequired[PairTrace]
    pair_ref: NotRequired[PairReference]
    pair_diagnostics: NotRequired[ConservationDiagnostics]
    audits: NotRequired[list[dict[str, Any]] | None]
    pair_frames_fingerprint: NotRequired[str]


class ResolvedMicroResult(_MicroIdentity):
    """Read-only-file view after resolving the shared pair into memory."""

    frames: list[MicroFrame]
    pair: PairTrace
    pair_ref: NotRequired[PairReference]
    pair_diagnostics: NotRequired[ConservationDiagnostics]
    audits: NotRequired[list[dict[str, Any]] | None]
    pair_frames_fingerprint: NotRequired[str]


class AdapterResult(_MicroIdentity):
    """Transport includes nil micro frames on pair_only and an empty pair on micro_only."""

    frames: list[MicroFrame] | None
    pair: NotRequired[dict[str, Any]]
    audits: NotRequired[list[dict[str, Any]] | None]
    pair_ref: NotRequired[PairReference]


MicroRecord: TypeAlias = Record[MicroTask, MicroResult]
ResolvedMicroRecord: TypeAlias = Record[MicroTask, ResolvedMicroResult]
PairRecord: TypeAlias = Record[PairTask, SharedPairResult]


class MeasureResult(TypedDict):
    series: dict[str, NumericArray]
    diagnostics: ConservationDiagnostics
    request_id: NotRequired[str]
    summary: NotRequired[dict[str, Any]]
    snapshots: NotRequired[dict[str, NumericArray]]


class PopulationPolicies(TypedDict):
    initial_rng: Literal["shared", "split"]
    projection: Literal["fixed_degree_rows", "actual"]
    record_policy: Literal["audit", "population"]
    energy_normalization: Literal["nominal", "actual"]


class PopulationRequest(PopulationPolicies):
    """The public binding owns the encoded nested kinetic request schema."""

    id: str
    kind: Literal["micro", "micro_only", "pair_only"]
    request: dict[str, Any]
    init_seed: int
    process_seed: int
    selection: Selection
    network: Literal["fixed"]
    physical_dt: float
    snapshot_steps: list[int]
    budgets: bool
    microscopic_initial: dict[str, Any]
    micro_options: dict[str, int | float]
    compact_records: bool


class ReleaseProtocol(TypedDict):
    schema: int
    bootstrap_seed: int
    bootstrap_samples: int
    confidence_level: float
    plateau_window_tau: float
    minimum_plateau_tau: float
    plateau_relative_tolerance: float
    prefix_taus: list[float]
    primary_clock: str
    primary_reference: str
    uncertainty_unit: str
    interpretation: str
