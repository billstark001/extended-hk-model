"""Finite-exposure mesoscopic theory and observable definitions.

Arrays represent cell masses: sum(rho)=1, sum_j E_ij=k*rho_i.
Numerical time integration belongs to the external Go runtime.
"""

from .energy import edge_energy, interaction_matrix, node_energy
from .release import release_clock
from .rewiring import replacement_flux
from .state import PairState, opinion_grid

__all__ = [
    "PairState",
    "opinion_grid",
    "interaction_matrix",
    "node_energy",
    "edge_energy",
    "release_clock",
    "replacement_flux",
]
