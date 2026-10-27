"""Study-owned frozen plans and calibrated run budgets."""

from typing import Literal

from typing_extensions import TypedDict

from lifted.types import Task


class Stage(TypedDict):
    name: str
    task_ids: list[str]


class Plan(TypedDict):
    schema: Literal[1]
    profile: Literal["smoke", "pilot", "formal"]
    tasks: list[Task]
    stages: list[Stage]
    protocol: dict
    sources: dict[str, str]
    budget: dict


class FrozenPlan(Plan):
    plan_fingerprint: str
