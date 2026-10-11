"""Facade for the original evidence plan and the current main-figure plan."""

from .paper_design import KERNELS, PROTOCOL, paper_design


def design(profile="paper", horizon=None):
    """Dispatch a named profile without embedding an experiment design here."""
    if profile in ("main", "main_smoke"):
        from .main_design import design as main_design

        return main_design(smoke=profile == "main_smoke", horizon=horizon)
    return paper_design(profile=profile, horizon=horizon)
