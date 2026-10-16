"""Short software exercises for each retained topic; never manuscript evidence."""

from ..types import MainPlan, MainProfile, MainTopic
from .main_design import TOPICS, StudyDesign, StudySettings


def design(
    profile: MainProfile = "main_smoke", topics: tuple[MainTopic, ...] = TOPICS, horizon: float | None = None
) -> MainPlan:
    horizon = 0.4 if horizon is None else horizon
    if not 0 < horizon <= 1:
        raise ValueError("main smoke horizon must be positive and <=1")
    settings = StudySettings(
        horizon=horizon,
        bins=21,
        quadrature=3,
        record_tau=0.02,
        epsilons=(0.4,),
        snapshot_taus=(0.1, 0.2),
        rates=(0.0, 0.25, 2.0, 4.0),
        scaling_ratios=(2.0,),
        alphas=(0.01, 0.02),
        controls=((21, 3, 1.0), (31, 3, 1.0), (21, 5, 1.0), (21, 3, 0.5)),
        populations=(40, 80),
        population_kernel_indices=(0, 1, 2, 3, 4),
        population_horizon=horizon,
        population_initials=2,
        population_branches=2,
        matched_population=40,
        matched_branches=2,
        adaptive_stopping=False,
        prefix_minimum=0.1,
        main_snapshot_tau=min(0.2, horizon),
    )
    return StudyDesign(settings).freeze(profile, topics)
