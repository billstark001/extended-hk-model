"""Clipped quadratic energy; do not multiply cell masses by dx again."""

import numpy as np


def interaction_matrix(axis, epsilon: float) -> np.ndarray:
    """Return clipped quadratic pair costs 0.5*min(distance**2, epsilon**2) on the supplied axis."""
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or not np.isfinite(axis).all() or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite axis and positive confidence radius required")
    return 0.5 * np.minimum((axis[:, None] - axis[None, :]) ** 2, epsilon**2)


def node_energy(rho, interaction):
    """Evaluate 0.5*rho.T@interaction@rho on cell masses, retaining any leading batch dimensions."""
    rho = np.asarray(rho, dtype=float)
    return 0.5 * np.einsum("...i,ij,...j->...", rho, interaction, rho)


def edge_energy(edge, interaction, degree: int):
    """Evaluate sum(edge*interaction)/degree on directed-edge masses, retaining leading batch dimensions."""
    if degree <= 0:
        raise ValueError("degree must be positive")
    return np.einsum("...ij,ij->...", edge, interaction) / degree


def potentials(state, interaction):
    """Return node and conditional edge potentials by source cell; edge potentials are NaN on empty source cells."""
    node = interaction @ state.rho
    edge = np.divide(
        np.sum(state.edge * interaction, axis=1),
        state.degree * state.rho,
        out=np.full_like(state.rho, np.nan),
        where=state.rho > 0,
    )
    return node, edge


def discordant_edge_mass(state, confidence):
    """Fraction of edges outside confidence, using the supplied geometry."""
    return float(np.sum(state.edge * (1 - np.asarray(confidence))) / state.degree)
