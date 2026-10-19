"""Central study-owned frozen-plan and run metadata types; no design defaults.

Library physics/record types live in theory.types and generic storage/build
contracts in common.types. Declaring a schema does not rewrite frozen data.
"""

from typing import Literal, TypeAlias

from typing_extensions import NotRequired, TypedDict

from common.types import Backend, RuntimeBinding, ScientificSources
from theory.types import ReleaseProtocol, Task

MainTopic: TypeAlias = Literal["recommendation", "scaling", "numerics", "matched", "population", "network_snapshot"]
MainProfile: TypeAlias = Literal[
    "main",
    "main_smoke",
    "recommendation",
    "scaling",
    "numerics",
    "matched",
    "population",
    "smoke_recommendation",
    "smoke_scaling",
    "smoke_numerics",
    "smoke_matched",
    "smoke_population",
    "network_snapshot",
    # Frozen pre-rename profiles retain their original bytes and identities.
    "main_recommendation",
    "main_smoke_recommendation",
    "main_scaling",
    "main_smoke_scaling",
    "main_numerics",
    "main_smoke_numerics",
    "main_matched",
    "main_smoke_matched",
    "main_population",
    "main_smoke_population",
]
MainRole: TypeAlias = Literal["effects", "scaling", "numerics", "closure", "matched", "population", "network_snapshot"]


class NumericalBaseline(TypedDict):
    bins: int
    quadrature: int
    dt: float


class AnalysisProtocol(ReleaseProtocol):
    numerical_baseline: NumericalBaseline
    main_population_bins: int
    main_population_quadrature: int
    population_horizon_tau: float
    population_kernels: list[list[str | float]]
    population_ratio: float
    population_claim: str
    main_snapshot_tau: float
    baseline_alpha: float
    baseline_epsilon: float
    confidence_radii: list[float]
    matched_population: int
    matched_bins: int
    matched_quadrature: int
    adaptive_measure_stopping: bool
    maximum_horizon_tau: float
    measure_stopping: str


class MainPlan(TypedDict):
    schema: Literal[1]
    profile: MainProfile
    analysis_protocol: AnalysisProtocol
    tasks: list[Task]


class FrozenMainPlan(MainPlan):
    plan_fingerprint: str


class RunManifest(TypedDict):
    schema: Literal[1]
    work_dir: str
    publish_dir: str
    profile: MainProfile
    horizon: float | None
    backend: Backend
    sources: ScientificSources
    request_schema_sha256: str


class CoverageRow(TypedDict):
    id: str
    kind: Literal["measure", "micro"]
    roles: str
    complete: bool


class ProvenanceRow(TypedDict):
    id: str
    sha256: str
    path: str


class EvidenceReport(TypedDict):
    schema: Literal[1]
    profile: MainProfile
    plan_fingerprint: str
    planned: int
    completed: int
    missing: list[str]
    measure: int
    early_stopped_measure: int
    observed_measure_horizon_range: list[float]
    micro: int
    conditional_cohorts: int
    confirmed_measure_rows: int
    analysis_protocol: AnalysisProtocol
    bindings: dict[str, RuntimeBinding]
    initial_and_process_controls: str
    claim_scope: str
    source_scope: NotRequired[str]
