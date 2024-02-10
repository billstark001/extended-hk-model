from typing import Dict, Union
import numpy as np
import networkx as nx
from numpy.typing import NDArray

class NetworkLayoutCollector:
  
  def collect(self, digraph: nx.DiGraph, graph: nx.Graph, n: int, opinion: NDArray) -> float:
    
    pos = nx.spring_layout(digraph)
    return pos, opinion, nx.DiGraph(digraph)