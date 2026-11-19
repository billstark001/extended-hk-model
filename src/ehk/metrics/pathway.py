"""State-based pathway observables and first observed passage."""

from collections.abc import Iterable
from dataclasses import dataclass

SEPARATION_THRESHOLD = 0.4
EXCESS_THRESHOLD = 0.1
OBSERVATION_INTERVAL = 16
PATHWAY_LABELS = ("PbS", "SbP", "simultaneous", "separation_only", "segregation_only", "neither", "unavailable")


@dataclass(frozen=True)
class FirstPassage:
    separation_step: int | None
    segregation_step: int | None
    complete: bool = True

    @property
    def label(self) -> str:
        if not self.complete:
            return "unavailable"
        p, s = self.separation_step, self.segregation_step
        if p is None:
            return "neither" if s is None else "segregation_only"
        if s is None:
            return "separation_only"
        if p == s:
            return "simultaneous"
        return "PbS" if p < s else "SbP"


def first_passage(states: Iterable[tuple[int, float | None, float | None]]) -> FirstPassage:
    """Use the first recorded crossing of each line, with no interpolation."""
    separation_step = segregation_step = None
    complete = False
    gap = False
    for step, separation, excess in states:
        complete = True
        if (separation_step is None and separation is None) or (segregation_step is None and excess is None):
            gap = True
        if separation_step is None and separation is not None and separation >= SEPARATION_THRESHOLD:
            separation_step = step
        if segregation_step is None and excess is not None and excess >= EXCESS_THRESHOLD:
            segregation_step = step
        if separation_step is not None and segregation_step is not None:
            break
    return FirstPassage(separation_step, segregation_step, complete and not gap)
