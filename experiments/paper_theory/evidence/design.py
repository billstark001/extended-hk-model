"""Dispatch only the current main and main_smoke designs."""

from . import main_design


def design(profile="main", horizon=None):
    if profile not in ("main", "main_smoke"):
        raise ValueError("profile must be main or main_smoke")
    return main_design.design(smoke=profile == "main_smoke", horizon=horizon)
