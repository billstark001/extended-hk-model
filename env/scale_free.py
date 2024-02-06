from typing import Tuple, Optional
from numpy.typing import NDArray

import dataclasses

import networkx as nx
import numpy as np


def preferential_attachment(seq, m: int, rng: np.random.RandomState):
  """Return m unique elements from seq.

  This differs from random.sample which can return repeated
  elements if seq holds repeated elements.

  Note: rng is a random.Random or numpy.random.RandomState instance.
  """
  targets = set()
  while len(targets) < m:
    x = rng.choice(seq)
    targets.add(x)
  return targets


def barabasi_albert_digraph_inplace(
    n: int,
    m: int,
    G: nx.DiGraph,
    p: Optional[float] = None,
    seed: Optional[int] = None
):

  # sanity check
  if m < 1 or m >= n:
    raise nx.NetworkXError(
        f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
    )
  if len(G) < m or len(G) > n:
    raise nx.NetworkXError(
        f"Barabási–Albert initial graph needs between m={m} and n={n} nodes"
    )

  # random state
  rng = np.random.RandomState(seed)

  # List of existing nodes, with nodes repeated once for each adjacent edge
  repeated_nodes = [n for n, d in G.degree() for _ in range(d)]
  # Start adding the other n - m0 nodes.
  source = len(G)
  while source < n:

    # Now choose m unique nodes from the existing nodes
    # Pick uniformly from repeated_nodes (preferential attachment)
    targets = preferential_attachment(repeated_nodes, m, rng)
    # Add edges to m nodes from the source.
    G.add_edges_from(zip([source] * m, targets))
    # Add one node to the list for each new edge just created.
    repeated_nodes.extend(targets)
    # And the new node "source" has m edges to add to the list.
    repeated_nodes.extend([source] * m)

    # handle closure with Holme-Kim model
    if p is not None:
      # triad closure for p
      exec_closure_array = rng.rand(len(targets)) < p
      # add m edges
      for target, closure in zip(targets, exec_closure_array):
        G.add_edge(source, target)

        if not closure:
          continue

        neighbors = list(G.neighbors(target))
        if neighbors:
          w = rng.choice(neighbors)
          if not G.has_edge(source, w):
            G.add_edge(source, w)

    # update source
    source += 1

  return G


@dataclasses.dataclass
class ScaleFreeNetworkProvider:

  seed: Optional[int] = None

  agent_count: int = 1000
  agent_follow: int = 20

  agent_closure: Optional[float] = None
  init_agent_count: int = 30

  opinion_range: Tuple[int, int] = (-1, 1)

  def generate(self) -> Tuple[nx.DiGraph, NDArray]:
    graph: nx.DiGraph = nx.erdos_renyi_graph(
        n=self.init_agent_count,
        p=self.agent_follow / (self.init_agent_count - 1),
        seed=None if self.seed is None else self.seed + 1,
        directed=True
    )
    barabasi_albert_digraph_inplace(
        n=self.agent_count,
        m=self.agent_follow,
        p=self.agent_closure,
        G=graph,
        seed=self.seed,
    )
    opinion = np.random.uniform(*self.opinion_range, (self.agent_count, ))
    return graph, opinion
