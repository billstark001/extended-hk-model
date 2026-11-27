"""One immutable extraction of an opinion state and its ordered directed edges."""

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from smp_bindings.record import RawSimulationRecord


@dataclass
class MeasurementRecord:
    """Saved numeric history and its graph reader, detached from wire metadata."""

    opinions: NDArray | Sequence[NDArray]
    max_step: int
    epsilon: float
    graph_at: Callable[[int], nx.DiGraph]
    directory: str | None = None

    @classmethod
    def from_saved_record(cls, record: "RawSimulationRecord") -> "MeasurementRecord":
        """Read the external SMP record once at the ingestion boundary."""
        return cls(
            opinions=record.opinions,
            max_step=record.max_step,
            epsilon=float(record.metadata.HKParams.Tolerance),
            graph_at=record.get_graph,
            directory=record.directory,
        )


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

