from typing import List, Dict, Optional, Callable, Any
import time
import numpy as np
from tqdm import tqdm
import networkx as nx

from ehk_model_old.base import HKAgent, HKModel, HKModelRecommendationSystem


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
      
      steepness: Optional[float] = None, # None == \infty
      noise_std: float = 0.5,
      random_ratio: float = 0,
      
      matrix_init: bool = False,
      log: Optional[Callable[[str], None]] = None,
  ):
    super().__init__(model)
    
    self.steepness = steepness
    self.noise_std = noise_std
    self.random_ratio = random_ratio
    
    self.matrix_init = matrix_init
    self.log = log
    
    # placeholders
    self.rate_mat = self.conn_mat = self.all_indices = np.zeros((0, 0))
    self.agent_map: Dict[int, HKAgent] = {}
    self.num_nodes = 0
    
  def dump(self) -> Any:
    return self.conn_mat

  def post_init(self, dump_data: Optional[Any] = None):
    self.num_nodes = n = self.model.graph.number_of_nodes()
    self.all_indices = np.arange(self.num_nodes, dtype=int)
    
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
      
    

  def pre_step(self):
    
    raw_rate_mat = self.conn_mat + self.conn_mat.T
    
    if self.noise_std > 0:
      noise_mat = np.random.normal(0, self.noise_std, raw_rate_mat.shape)
      raw_rate_mat = raw_rate_mat * (1 - 2 * noise_mat) + noise_mat
      raw_rate_mat[raw_rate_mat < 0] = 0
      
    np.fill_diagonal(raw_rate_mat, 0)
    
    if self.steepness is not None and self.steepness != 1:
      raw_rate_mat = raw_rate_mat ** self.steepness
      
    # if steepness is not infinity, normalize the rate matrix
    if self.steepness is not None:
      rate_sum_rev = np.reshape(np.sum(raw_rate_mat, axis=1), (-1, 1)) ** -1
      rate_mat = raw_rate_mat * rate_sum_rev
      if self.random_ratio > 0:
        rate_mat = (1 - self.random_ratio) * rate_mat + self.random_ratio / (self.num_nodes - 1)
      np.fill_diagonal(rate_mat, 0)
    else:
      rate_mat = raw_rate_mat
      
    # expose rate matrix
    self.rate_mat = rate_mat
    

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    neighbor_ids = np.array([x.unique_id for x in neighbors + [agent]], dtype=int)
    raw_rate_vec = self.rate_mat[agent.unique_id]

    ret: np.ndarray
    if self.steepness is None:
      ret = np.setdiff1d(np.argpartition(
          raw_rate_vec, len(neighbors) + count), neighbor_ids)
    else:
      rate_vec = np.copy(raw_rate_vec)
      rate_vec[neighbor_ids] = 0
      rate_vec /= np.sum(rate_vec)
      ret = np.random.choice(self.all_indices, (count,), replace=False, p=rate_vec)

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

