"""Paired edge replacement in the old-opinion frame.

This is the mean signed edge increment per opportunity. Transport and the
synchronous mixed term are not included in this frozen-state operator.
"""

from dataclasses import dataclass

import numpy as np

from .exposure import at_least_one
from .state import PairState


@dataclass(frozen=True)
class Replacement:
    """Frozen-state replacement flux and its conditional components.

    flux/loss/gain have shape (bins, bins); eligibility, discordance and concordance
    are source-cell vectors. flux is per opportunity, before multiplication by q.
    """

    flux: np.ndarray
    eligibility: np.ndarray
    discordance: np.ndarray
    concordance: np.ndarray
    loss: np.ndarray
    gain: np.ndarray


def replacement_flux(state: PairState, recommendation, confidence, recommendation_count: int) -> Replacement:
    """J_ij=rho_i h_i(g_ij-ell_ij); the evolution rate is q*J."""
    rec = np.asarray(recommendation, dtype=float)
    mask = np.asarray(confidence, dtype=float)
    if rec.shape != state.edge.shape or mask.shape != state.edge.shape:
        raise ValueError("recommendation and confidence must match edge shape")
    if not np.isfinite(rec).all() or np.any(rec < 0) or not np.allclose(rec.sum(1), 1):
        raise ValueError("recommendation rows must be probability masses")
    if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError("confidence weights must lie in [0,1]")
    loss = state.neighbors() * (1 - mask)
    gain = rec * mask
    d = np.clip(loss.sum(1), 0, 1)
    c = np.clip(gain.sum(1), 0, 1)
    loss = np.divide(loss, d[:, None], out=np.zeros_like(loss), where=d[:, None] > 0)
    gain = np.divide(gain, c[:, None], out=np.zeros_like(gain), where=c[:, None] > 0)
    eligibility = at_least_one(d, state.degree) * at_least_one(c, recommendation_count)
    flux = state.rho[:, None] * eligibility[:, None] * (gain - loss)
    return Replacement(flux, eligibility, d, c, loss, gain)


def energy_rate(replacement: Replacement, interaction, degree: int, q: float):
    """Return the signed frozen-state edge-energy rate q*sum(flux*interaction)/degree; transport is excluded."""
    if degree <= 0 or not np.isfinite(q) or q < 0:
        raise ValueError("positive degree and nonnegative finite q required")
    return float(q * np.sum(replacement.flux * interaction) / degree)


def frozen_discordance_rate(replacement: Replacement, degree: int, q: float):
    """d'_i=-(q/k) h_i when the confidence geometry is fixed."""
    if degree <= 0 or not np.isfinite(q) or q < 0:
        raise ValueError("positive degree and nonnegative finite q required")
    return -q / degree * replacement.eligibility
