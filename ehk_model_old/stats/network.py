from typing import Optional, Mapping
import numpy as np
import networkx as nx
from numpy.typing import NDArray


class NetworkLayoutCollector:

  def __init__(self, use_last=False, return_dict=False, use_pos=False):
    self.last: Optional[Mapping] = None
    self.use_last = use_last
    self.return_dict = return_dict
    self.use_pos = use_pos

  def collect(
          self,
          prefix: str,
          n: int, step: int,
          digraph: nx.DiGraph, graph: nx.Graph, opinion: NDArray,
          *args, **kwargs) -> float:

    if step <= 0:
      self.last = None

    pos = nx.spring_layout(digraph, pos=self.last) \
      if self.use_pos else None
    if self.use_last:
      self.last = pos
      
    # get a clear digraph
    graph = nx.DiGraph(digraph)
    for n in graph:
      del graph.nodes[n]['agent']

    if self.return_dict:
      ret = {
          prefix + '-opinion': opinion,
          prefix + '-graph': graph,
      }
      if self.use_pos:
        ret[prefix + '-pos'] = pos
      return ret

    return pos, opinion, graph
