from typing import Optional, Mapping
import numpy as np
import networkx as nx
from numpy.typing import NDArray


class NetworkLayoutCollector:

  def __init__(self, use_last=False, return_dict=False):
    self.last: Optional[Mapping] = None
    self.use_last = use_last
    self.return_dict = return_dict

  def collect(
          self,
          prefix: str,
          n: int, step: int,
          digraph: nx.DiGraph, graph: nx.Graph, opinion: NDArray,
          *args, **kwargs) -> float:

    if step <= 0:
      self.last = None

    pos = nx.spring_layout(digraph, pos=self.last)
    if self.use_last:
      self.last = pos
    graph = nx.DiGraph(digraph)
    for n in graph:
      del graph.nodes[n]['agent']

    if self.return_dict:
      return {
          prefix + '-pos': pos,
          prefix + '-opinion': opinion,
          prefix + '-graph': graph,
      }

    return pos, opinion, graph
