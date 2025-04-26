from typing import Dict, Union
import numpy as np
import networkx as nx
from numpy.typing import NDArray


class TriadsCountCollector:

  def collect(self, graph: nx.Graph, *args, **kwargs) -> float:

    triads = nx.triangles(graph)
    triads_count = sum(triads.values()) // 3

    return triads_count


class ClusteringCollector:

  def collect(self, graph: nx.Graph, *args, **kwargs) -> float:

    clustering = nx.average_clustering(graph)
    return clustering


class SegregationIndexCollector:

  def collect(
          self,
          n: int, graph: nx.Graph, opinion: NDArray,
          *args, **kwargs) -> float:

    opinion_avg = np.mean(opinion)
    _opinion = opinion - opinion_avg

    positive_amount = max(1, np.sum(_opinion > 0))
    negative_amount = max(1, n - positive_amount)
    edge_interconnection = len(
        [None for u, v in graph.edges if _opinion[u] * _opinion[v] <= 0])

    density = graph.number_of_edges() / (n * (n - 1) / 2)
    s_index: float = 1 - edge_interconnection / \
        (2 * density * positive_amount * negative_amount)

    return s_index
