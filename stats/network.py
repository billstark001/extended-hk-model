from typing import Optional, Mapping
import numpy as np
import networkx as nx
from numpy.typing import NDArray

class NetworkLayoutCollector:
  
  def __init__(self, use_last=False):
    self.last: Optional[Mapping] = None
    self.use_last = use_last
  
  def collect(self, digraph: nx.DiGraph, graph: nx.Graph, n: int, opinion: NDArray) -> float:
    
    pos = nx.spring_layout(digraph, pos=self.last)
    if self.use_last:
      self.last = pos
    return pos, opinion, nx.DiGraph(digraph)