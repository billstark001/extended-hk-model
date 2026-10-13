"""Dispatch only the current main and main_smoke designs."""

from ..types import MainPlan, MainProfile
from . import main_design


def design(profile: MainProfile = "main", horizon: float | None = None) -> MainPlan:
    if profile not in ("main", "main_smoke"):
        raise ValueError("profile must be main or main_smoke")
    return main_design.design(smoke=profile == "main_smoke", horizon=horizon)
