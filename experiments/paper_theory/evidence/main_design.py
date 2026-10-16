"""Formal manuscript designs: recommendation, scaling, controls and two ensembles."""

from dataclasses import asdict, dataclass, replace
from itertools import product
from math import ceil
from typing import cast

from typing_extensions import Unpack

from common.io.json import fingerprint
from theory.parameters import Parameters
from theory.types import (
    MeasureOptions,
    MicroTaskFields,
    ParametersData,
    StoppingOptions,
    Task,
    TaskKind,
    TaskSpecification,
)

from ..types import AnalysisProtocol, MainPlan, MainProfile, MainRole, MainTopic
from .protocol import KERNELS, PROTOCOL

TOPICS: tuple[MainTopic, ...] = ("recommendation", "scaling", "numerics", "matched", "population")


@dataclass(frozen=True)
class StudySettings:
    """Physical and sampling choices; the software exercise supplies its own settings."""

    horizon: float = 480.0
    bins: int = 161
    quadrature: int = 33
    record_tau: float = 0.1
    epsilons: tuple[float, ...] = (0.2, 0.4, 0.8)
    snapshot_taus: tuple[float, ...] = (4.0, 24.0, 30.0, 120.0, 240.0, 480.0)
    rates: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 16.0, 32.0)
    scaling_ratios: tuple[float, ...] = (0.25, 2.0, 10.0)
    alphas: tuple[float, ...] = (0.04, 0.02, 0.01, 0.005, 0.0025)
    controls: tuple[tuple[int, int, float], ...] = (
        (161, 33, 1.0),
        (81, 33, 1.0),
        (161, 9, 1.0),
        (161, 33, 0.5),
        (241, 33, 1.0),
    )
    populations: tuple[int, ...] = (250, 500, 1000, 2000)
    population_kernel_indices: tuple[int, ...] = (0, 2)
    population_horizon: float = 120.0
    population_initials: int = 4
    population_branches: int = 4
    matched_population: int = 500
    matched_branches: int = 8
    adaptive_stopping: bool = True
    prefix_minimum: float = 30.0
    main_snapshot_tau: float = 24.0


def stopping_options(p: Parameters) -> StoppingOptions:
    """Check joint state/energy stationarity in scaled time, independent of dt/alpha."""
    tau_step = p.alpha * p.dt
    return {
        "mode": "state_and_energy",
        "minimum_steps": ceil(PROTOCOL["minimum_plateau_tau"] / tau_step),
        "check_every": max(1, ceil(0.2 / tau_step)),
        "patience_steps": ceil(2 * PROTOCOL["plateau_window_tau"] / tau_step),
        "state_l1_tolerance": 1e-5 * tau_step,
        "energy_absolute_tolerance": 1e-7 * tau_step,
        "energy_relative_tolerance": 1e-5 * tau_step,
    }


class StudyDesign:
    """Assemble the selected scientific topics and deduplicate shared trajectories."""

    def __init__(self, settings: StudySettings):
        self.settings = settings
        self.base = Parameters(
            alpha=0.02,
            epsilon=0.4,
            bins=settings.bins,
            quadrature=settings.quadrature,
            horizon=settings.horizon,
            record_tau=settings.record_tau,
        )
        self.snapshots = sorted({min(t, settings.horizon) for t in (*settings.snapshot_taus, settings.horizon)})
        self.tasks: dict[str, Task] = {}

    def add(self, kind: TaskKind, p: Parameters, role: MainRole, **extra: Unpack[MicroTaskFields]) -> None:
        for epsilon in self.settings.epsilons:
            setting = replace(p, epsilon=epsilon)
            setting.validate()
            options: MeasureOptions = {"snapshot_taus": self.snapshots}
            if kind == "measure" and self.settings.adaptive_stopping:
                options["stopping"] = stopping_options(setting)
            task: TaskSpecification = dict(
                kind=kind, parameters=cast(ParametersData, asdict(setting)), request_options=options, **extra
            )
            identifier = kind + "_" + fingerprint(task)[:20]
            if identifier not in self.tasks:
                self.tasks[identifier] = cast(Task, dict(task, id=identifier, roles=[]))
            if role not in self.tasks[identifier]["roles"]:
                self.tasks[identifier]["roles"].append(role)

    def recommendation(self) -> None:
        for rule, (rec, z), ratio in product(("hk", "deffuant"), KERNELS, self.settings.rates):
            self.add("measure", replace(self.base, dynamics=rule, recommender=rec, steepness=z, ratio=ratio), "effects")

    def scaling(self) -> None:
        for rule, (rec, z), ratio, alpha in product(
            ("hk", "deffuant"), KERNELS, self.settings.scaling_ratios, self.settings.alphas
        ):
            self.add(
                "measure",
                replace(self.base, dynamics=rule, recommender=rec, steepness=z, ratio=ratio, alpha=alpha),
                "scaling",
            )

    def numerics(self) -> None:
        for rule, (rec, z), (bins, quad, dt) in product(
            ("hk", "deffuant"), (KERNELS[2], KERNELS[4]), self.settings.controls
        ):
            self.add(
                "measure",
                replace(self.base, dynamics=rule, recommender=rec, steepness=z, bins=bins, quadrature=quad, dt=dt),
                "numerics",
            )

    def population(self) -> None:
        s = self.settings
        for rule, index, n, initial, branch in product(
            ("hk", "deffuant"),
            s.population_kernel_indices,
            s.populations,
            range(s.population_initials),
            range(s.population_branches),
        ):
            rec, z = KERNELS[index]
            self.add(
                "micro",
                replace(
                    self.base, dynamics=rule, recommender=rec, steepness=z, population=n, horizon=s.population_horizon
                ),
                "population",
                initial=initial,
                branch=branch,
                initial_seed=17000 + initial,
                process_seed=24000 + initial * 100 + branch,
                selection="weighted",
            )

    def matched(self) -> None:
        for rule, (rec, z), initial, branch in product(
            ("hk", "deffuant"), (KERNELS[0], KERNELS[2], KERNELS[4]), range(2), range(self.settings.matched_branches)
        ):
            self.add(
                "micro",
                replace(
                    self.base, dynamics=rule, recommender=rec, steepness=z, population=self.settings.matched_population
                ),
                "matched",
                initial=initial,
                branch=branch,
                initial_seed=17000 + initial,
                process_seed=24000 + initial * 100 + branch,
                selection="weighted",
            )

    def freeze(self, profile: MainProfile, topics: tuple[MainTopic, ...]) -> MainPlan:
        # Preserve the original union's order, including overlapping smoke cohorts.
        for topic in ("recommendation", "scaling", "numerics", "population", "matched"):
            if topic in topics:
                getattr(self, topic)()
        s = self.settings
        protocol: AnalysisProtocol = {
            **PROTOCOL,
            "prefix_taus": [t for t in self.snapshots if t >= s.prefix_minimum] or [s.horizon],
            "numerical_baseline": {"bins": s.bins, "quadrature": s.quadrature, "dt": 1.0},
            "main_population_bins": s.bins,
            "main_population_quadrature": s.quadrature,
            "population_horizon_tau": s.population_horizon,
            "population_kernels": [list(KERNELS[i]) for i in s.population_kernel_indices],
            "population_ratio": 2.0,
            "population_claim": "finite-window structure robustness; structural-score size audit is separate",
            "main_snapshot_tau": s.main_snapshot_tau,
            "baseline_alpha": 0.02,
            "baseline_epsilon": 0.4,
            "confidence_radii": list(s.epsilons),
            "matched_population": s.matched_population,
            "matched_bins": s.bins,
            "matched_quadrature": s.quadrature,
            "minimum_plateau_tau": 120.0,
            "adaptive_measure_stopping": s.adaptive_stopping,
            "maximum_horizon_tau": s.horizon,
            "measure_stopping": "joint state/energy rates; tau>=120, stable for 40 tau; no extrapolated tail",
            "primary_clock": "unclipped finite-window residual area; full means require tail confirmation",
        }
        return dict(schema=1, profile=profile, analysis_protocol=protocol, tasks=list(self.tasks.values()))


def design(
    profile: MainProfile = "main", topics: tuple[MainTopic, ...] = TOPICS, horizon: float | None = None
) -> MainPlan:
    horizon = 480.0 if horizon is None else horizon
    if not 480 <= horizon < float("inf"):
        raise ValueError("formal main horizon must be finite and >=480")
    return StudyDesign(StudySettings(horizon=horizon)).freeze(profile, topics)
