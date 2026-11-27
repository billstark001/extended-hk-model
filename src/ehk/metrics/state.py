"""One immutable extraction of an opinion state and its ordered directed edges."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _readonly(values):
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class MeasurementState:
    """Share topology extraction and edge differences across distance and homophily.

    Opinion dtype and directed-edge order are retained because both belong to
    the historical KDE sampling convention. Confidence is a measurement
    parameter, not part of the physical state.
    """

    opinions: NDArray
    edges: NDArray
    edge_distances: NDArray
    out_degree: NDArray

    @classmethod
    def from_graph(cls, graph, opinions):
        return cls.from_edges(opinions, np.asarray(list(graph.edges), dtype=int).reshape(-1, 2))

    @classmethod
    def from_edges(cls, opinions, edges):
        values = np.asarray(opinions)
        neighbors = np.asarray(edges)
        if values.ndim != 1:
            raise ValueError("opinions must be a one-dimensional array")
        if neighbors.size == 0:
            neighbors = np.empty((0, 2), dtype=int)
        if neighbors.ndim != 2 or neighbors.shape[1] != 2 or not np.issubdtype(neighbors.dtype, np.integer):
            raise ValueError("ordered edges must be an integer array of shape (m, 2)")
        neighbors = neighbors.astype(int, copy=False)
        if neighbors.size and (neighbors.min() < 0 or neighbors.max() >= len(values)):
            raise ValueError("edge endpoints must index the opinion array")
        differences = np.abs(values[neighbors[:, 1]] - values[neighbors[:, 0]])
        degree = np.bincount(neighbors[:, 0], minlength=len(values))
        return cls(_readonly(values), _readonly(neighbors), _readonly(differences), _readonly(degree))

