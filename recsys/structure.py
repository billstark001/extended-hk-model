from typing import List, Dict, Optional, Callable, Any
import time
import numpy as np
from tqdm import tqdm
import networkx as nx

from base import HKAgent, HKModel, HKModelRecommendationSystem


def common_neighbors_count(G: nx.DiGraph, u: int, v: int):
  i1 = len([w for w in G._pred[u] if w in G._pred[v] or w in G._succ[v]])
  i2 = len([w for w in G._succ[u] if w in G._pred[v] or w in G._succ[v]])
  return i1 + i2

short_progress_bar="{l_bar}{bar:10}{r_bar}{bar:-10b}"

class Structure(HKModelRecommendationSystem):

  num_nodes = 0

  def __init__(
      self,
      model: HKModel,
      eta: float = 1,
      sigma: float = 0.5,
      matrix_init: bool = False,
      log: Optional[Callable[[str], None]] = None,
  ):
    super().__init__(model)
    self.matrix_init = matrix_init
    self.log = log
    
    self.eta = eta if eta > 0 else 0
    self.sigma = sigma if sigma > 0 else -sigma
    self.agent_map: Dict[int, HKAgent] = {}
    
  def dump(self) -> Any:
    return self.conn_mat

  def post_init(self, dump_data: Optional[Any] = None):
    self.num_nodes = n = self.model.graph.number_of_nodes()
    
    # build agent map
    for a in self.model.schedule.agents:
      self.agent_map[a.unique_id] = a
    
    # load connection matrix if dumped
    if dump_data is not None:
      self.conn_mat = dump_data
      if self.log:
        self.log('Connection matrix loaded from dump data.')
      return
    
    # calculate full connection matrix 
    # recommend to use matrix calculation if n <= 1500
    
    tstart = time.time()
    if self.matrix_init:
      adj_mat = nx.to_numpy_array(self.model.graph, dtype=int)
      adj_mat += adj_mat.T
      self.conn_mat = np.array(adj_mat @ adj_mat)
    
    else:
      conn_mat = np.zeros((n, n), dtype=int)
      G = self.model.graph
      for u in tqdm(range(0, self.num_nodes), bar_format=short_progress_bar):
        for v in range(u + 1, self.num_nodes):  # v > u
          conn_mat[u, v] = common_neighbors_count(G, u, v)
      self.conn_mat = conn_mat
      
    # set irrelevant elements to 0
    self.conn_mat = np.triu(self.conn_mat)
    np.fill_diagonal(self.conn_mat, 0)
    
    
    tend = time.time()
    if self.log:
      self.log(f'Connection matrix generation costs {tend - tstart}s.')
      
    # placeholders
    self.epsilon_mat = np.zeros((0, 0))
    self.val_mat = np.zeros((0, 0))
    self.val_mat_raw = np.zeros((0, 0))
    

  def pre_step(self):
    self.val_mat_raw = self.conn_mat + self.conn_mat.T
    if self.sigma > 0:
      self.epsilon_mat = np.random.normal(0, self.sigma, (self.num_nodes, self.num_nodes))
      self.val_mat = self.val_mat_raw * (1 - 2 * self.epsilon_mat) + self.epsilon_mat
    else:
      self.val_mat = self.val_mat_raw
    self.val_mat[self.val_mat < 0] = 0
    if self.eta != 1:
      self.val_mat = self.val_mat ** self.eta
    pass

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    a = agent.unique_id
    ret1 = self.val_mat[a]

    exclude_ids = np.array([x.unique_id for x in neighbors + [agent]])
    ret = np.setdiff1d(np.argpartition(
        ret1, len(neighbors) + count), exclude_ids)

    return [self.agent_map[i] for i in ret[:count]]

  def post_step(self, changed: List[int]):
    # update connection matrix
    changed = list(set(changed))
    changed.sort()
    G = self.model.graph
    for i in range(len(changed)):
      for j in range(i + 1, len(changed)):
        u = changed[i]
        v = changed[j]
        self.conn_mat[u, v] = common_neighbors_count(G, u, v)
    pass

  def pre_commit(self):
    pass
