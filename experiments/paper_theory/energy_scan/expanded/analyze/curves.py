"""Tail-normalized descriptors for the supplied saved energy slots."""

from theory.release.descriptors import curve_descriptors

from .. import config
from ..storage import curve_slots


def slot_descriptors(slot):
    metadata = {k: slot[k] for k in ("dynamics", "epsilon", "alpha", "q", "q_over_alpha", "converged")}
    return [
        {
            **metadata,
            "energy": name,
            **curve_descriptors(
                slot["tau"], slot[name], slot["first_step_delta"][i], slot["alpha"], config.PROGRESS_LEVELS
            ),
        }
        for i, name in enumerate(config.ENERGY_NAMES)
    ]


def parameterize(shards):
    """Keep the historical descriptor entry point; no full scan is required."""
    return [row for slot in curve_slots(shards) for row in slot_descriptors(slot)]
