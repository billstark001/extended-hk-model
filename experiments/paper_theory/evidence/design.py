"""Named manuscript tasks and their separate software exercises."""

from typing import cast

from ..types import MainPlan, MainProfile, MainTopic
from . import main_design, main_smoke_design

PROFILES = ("main", "main_smoke", *main_design.TOPICS, *(f"smoke_{t}" for t in main_design.TOPICS))


LEGACY_PROFILES = tuple(f"main_{t}" for t in main_design.TOPICS) + tuple(f"main_smoke_{t}" for t in main_design.TOPICS)


def canonical_profile(profile: str) -> MainProfile:
    if profile in LEGACY_PROFILES:
        return cast(MainProfile, profile.removeprefix("main_"))
    return cast(MainProfile, profile)


def topics(profile: MainProfile) -> tuple[MainTopic, ...]:
    profile = canonical_profile(profile)
    if profile not in PROFILES:
        raise ValueError(f"unknown main task: {profile}; choose from {', '.join(PROFILES)}")
    if profile in ("main", "main_smoke"):
        return main_design.TOPICS
    return (cast(MainTopic, profile.removeprefix("smoke_")),)


def is_smoke(profile: MainProfile) -> bool:
    topics(profile)
    return profile == "main_smoke" or canonical_profile(profile).startswith("smoke_")


def design(profile: MainProfile = "main", horizon: float | None = None) -> MainPlan:
    selected = topics(profile)
    if is_smoke(profile):
        return main_smoke_design.design(profile, selected, horizon)
    return main_design.design(profile, selected, horizon)
