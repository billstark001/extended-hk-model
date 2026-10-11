"""Fixed-outdegree pair measures, with explicit mass conventions."""

from dataclasses import dataclass

import numpy as np


def opinion_grid(bins: int, minimum: float = -1.0, maximum: float = 1.0) -> np.ndarray:
    """Return uniform cell centers on [minimum, maximum], not bin edges."""
    if bins < 2 or not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        raise ValueError("a grid needs >=2 bins and a finite increasing domain")
    return minimum + (np.arange(bins) + 0.5) * (maximum - minimum) / bins


@dataclass(frozen=True)
class PairState:
    """Immutable node and directed-edge cell masses with fixed outgoing degree.

    rho has shape (bins,) and sums to one; edge has shape (bins, bins) and row sums
    degree*rho. Construction validates and copies both arrays, then makes them
    read-only. Neither array is a density and no extra cell width is applied.
    """

    rho: np.ndarray
    edge: np.ndarray
    degree: int

    def __post_init__(self):
        rho = np.array(self.rho, dtype=float, copy=True)
        edge = np.array(self.edge, dtype=float, copy=True)
        if self.degree < 1 or int(self.degree) != self.degree:
            raise ValueError("degree must be a positive integer")
        if rho.ndim != 1 or edge.shape != (rho.size, rho.size):
            raise ValueError("edge must have shape (len(rho), len(rho))")
        if not np.isfinite(rho).all() or not np.isfinite(edge).all():
            raise ValueError("state masses must be finite")
        if np.any(rho < 0) or np.any(edge < 0):
            raise ValueError("state masses must be nonnegative")
        if not np.isclose(rho.sum(), 1.0, atol=1e-10, rtol=0):
            raise ValueError("rho must sum to one; densities must be converted first")
        if not np.allclose(edge.sum(axis=1), self.degree * rho, atol=1e-10, rtol=0):
            raise ValueError("edge rows must sum to degree*rho")
        rho.setflags(write=False)
        edge.setflags(write=False)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "edge", edge)

    @classmethod
    def random_mixing(cls, rho, degree: int):
        """Construct independent-endpoint edge mass degree * outer(rho, rho)."""
        rho = np.asarray(rho, dtype=float)
        return cls(rho, degree * np.outer(rho, rho), degree)

    def neighbors(self) -> np.ndarray:
        """Return conditional target masses edge/(degree*rho); empty source rows use rho with zero flux contribution."""
        return np.divide(
            self.edge,
            (self.degree * self.rho)[:, None],
            out=np.broadcast_to(self.rho, self.edge.shape).copy(),
            where=self.rho[:, None] > 0,
        )
