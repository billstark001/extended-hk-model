"""Uniform-start channel coefficients and cell-mass field/energy formulas."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def opinion_grid(size: int = 161) -> np.ndarray:
    """Return uniform opinion-cell centers on [-1,1]."""
    dx = 2.0 / size
    return -1.0 + (np.arange(size) + 0.5) * dx


def interaction_matrix(x: np.ndarray, epsilon: float = 0.45) -> np.ndarray:
    """Return the clipped quadratic interaction matrix on opinion-cell centers."""
    distance = x[:, None] - x[None, :]
    return 0.5 * np.minimum(distance**2, epsilon**2)


def node_energy(rho: np.ndarray, interaction: np.ndarray) -> np.ndarray:
    """Return one node energy per time row of rho shaped (time, bins), without extra cell-width factors."""
    return 0.5 * np.einsum("ti,ij,tj->t", rho, interaction, rho, optimize=True)


def edge_energy(edge: np.ndarray, interaction: np.ndarray, degree: int = 15) -> np.ndarray:
    """Return edge energies for arrays shaped (time, bins, bins), normalized by the configured fixed degree."""
    return np.einsum("tij,ij->t", edge, interaction, optimize=True) / degree


def derived_fields(
    rho: np.ndarray, edge: np.ndarray, interaction: np.ndarray, degree: int = 15, mass_tolerance: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return node/edge potentials, node/edge energies and topology excess; empty edge rows have NaN potential."""
    phi_rho = rho @ interaction.T
    row_mass = edge.sum(axis=2)
    phi_edge = np.divide(
        np.einsum("tij,ij->ti", edge, interaction, optimize=True),
        row_mass,
        out=np.full_like(row_mass, np.nan),
        where=row_mass > mass_tolerance,
    )
    energy_rho = node_energy(rho, interaction)
    energy_edge = edge_energy(edge, interaction, degree)
    topology_excess = energy_edge - 2.0 * energy_rho
    return (phi_rho, phi_edge, energy_rho, energy_edge, topology_excess)


def continuum_alpha_coefficient(epsilon: float = 0.45, degree: int = 15, recommendations: int = 10) -> float:
    """Return -dot(U)/alpha at the uniform random-mixing start."""
    total_items = degree + recommendations

    def integrand(s: float) -> float:
        p = 0.5 * (epsilon + s)
        finite_list_factor = (1.0 - (1.0 - p) ** total_items) / p
        return finite_list_factor * (epsilon**2 - s**2) ** 2

    return quad(integrand, 0.0, epsilon, epsabs=1e-14)[0] / 16.0


def continuum_q_coefficient(epsilon: float = 0.45, degree: int = 15, recommendations: int = 10) -> float:
    """Return -dot(U_E)/q for random rewiring at the uniform start."""

    def eligibility(p: float) -> float:
        return (1.0 - p**degree) * (1.0 - (1.0 - p) ** recommendations)

    def boundary_integrand(s: float) -> float:
        p = 0.5 * (epsilon + s)
        gain_second_moment = (s * s - s * epsilon + epsilon * epsilon) / 3.0
        return eligibility(p) * (epsilon * epsilon - gain_second_moment)

    boundary = quad(boundary_integrand, 0.0, epsilon, epsabs=1e-14)[0]
    interior = (1.0 - epsilon) * eligibility(epsilon) * (2.0 * epsilon * epsilon / 3.0)
    return (boundary + interior) / (2.0 * degree)
