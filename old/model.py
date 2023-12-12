import networkx as nx
import numpy as np
import random
from typing import Dict, Optional, Callable, Iterable, List


def init_opinion(G: nx.graph):
  o = {node: random.uniform(-1, 1) for node in G.nodes()}
  return o


def sample_rec(
  G: nx.Graph,
  current: float,
  epsilon: float,
  target: float,
  count: int,
) -> List[int]:
  raise 'NIE' # TODO


def iterate_opinion(
  G: nx.Graph, 
  opinions: Dict[int, float], 
  epsilon: float = 0.25, 
  tolerance: float = 0.005,
  recsys: Optional[Callable[[
      float, # current opinion
      List[float], # neighbor opinion
      List[int], # close neighbor
      List[int], # not close neighbor
      float, # rate
    ], float]] = None,
  recsys_rate: float = 0.5,
  rewiring_rate: float = 0.1,
):
  new_opinions = {}
  max_diff = 0
  
  rewiring_set = []
  
  for x in G.nodes():
    ox = opinions[x]
    close_set = [n for n in G.neighbors(x) if abs(opinions[n] - ox) <= epsilon]
    not_close_set = [n for n in G.neighbors(x) if abs(opinions[n] - ox) > epsilon]
    neighbor_opinions = [opinions[n] for n in close_set]
    
    # call recommendation system
    samples: Optional[List[int]] = None
    if recsys is not None:
      target = recsys(ox, neighbor_opinions, close_set, not_close_set)
      count = int(recsys_rate * (len(close_set) + len(not_close_set)) + 0.5, recsys_rate)
      samples = sample_rec(G, ox, epsilon, target, count)
      close_set += samples
    
    if len(neighbor_opinions) > 0:
      o = np.mean(neighbor_opinions)
      max_diff = max(max_diff, abs(o - opinions[x]))
    else:
      o = opinions[x]
    new_opinions[x] = o

    if not_close_set and samples and rewiring_rate > 0:
      pass # TODO handle rewiring

  if max_diff < tolerance:
    return None # the model halts at convergence
  
  return new_opinions


