"""Scientific topic selection through the shared named-task registry."""

from typing import cast

from common.experiments.registry import TaskRegistry

from ..types import MainPlan, MainProfile, MainTopic
from . import main_design, main_smoke_design

REGISTRY = TaskRegistry(
    {**{t: (t, False) for t in main_design.TOPICS}, **{f"smoke_{t}": (t, True) for t in main_design.TOPICS}},
    {"main": main_design.TOPICS, "main_smoke": tuple(f"smoke_{t}" for t in main_design.TOPICS)},
)
PROFILES = REGISTRY.names


def canonical_profile(profile: str) -> MainProfile:
    """Interpret saved pre-rename metadata without registering CLI/code aliases."""
    legacy = {f"main_{t}" for t in main_design.TOPICS} | {f"main_smoke_{t}" for t in main_design.TOPICS}
    return cast(MainProfile, profile.removeprefix("main_") if profile in legacy else profile)


def topics(profile: str) -> tuple[MainTopic, ...]:
    return tuple(t for t, _ in REGISTRY.select(canonical_profile(profile)))


def is_smoke(profile: str) -> bool:
    return all(smoke for _, smoke in REGISTRY.select(canonical_profile(profile)))


def design(profile: MainProfile = "main", horizon: float | None = None) -> MainPlan:
    selected = REGISTRY.select(profile)
    module = main_smoke_design if all(smoke for _, smoke in selected) else main_design
    # The two modules are independently typed; avoid losing their callable type.
    if module is main_smoke_design:
        return main_smoke_design.design(profile, tuple(t for t, _ in selected), horizon)
    return main_design.design(profile, tuple(t for t, _ in selected), horizon)
