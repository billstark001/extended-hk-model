from typing import Tuple
from numpy.typing import NDArray

import dataclasses

import networkx as nx
import numpy as np

@dataclasses.dataclass
class RandomNetworkProvider:
  agent_count: int = 1000
  agent_follow: int = 20
  opinion_range: Tuple[int, int] = (-1, 1)

  def generate(self) -> Tuple[nx.DiGraph, NDArray]:
    graph: nx.DiGraph = nx.erdos_renyi_graph(
        n=self.agent_count,
        p=self.agent_follow / (self.agent_count - 1),
        directed=True
    )
    opinion = np.random.uniform(*self.opinion_range, (self.agent_count, ))
    return graph, opinion
