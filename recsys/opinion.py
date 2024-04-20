from typing import List, Dict, Optional, Any
import numpy as np
from base import HKAgent, HKModel, HKModelRecommendationSystem


class Opinion(HKModelRecommendationSystem):

  def __init__(self, model: HKModel, noise_std: float = 0.05):
    super().__init__(model)
    self.noise_std = noise_std
    self.num_nodes = 0
    self.epsilon = np.zeros((0, ))

  def post_init(self, dump_data: Optional[Any] = None):
    self.num_nodes = self.model.graph.number_of_nodes()
    self.agents: List[HKAgent] = list(self.model.schedule.agents)
    self.agent_indices: Dict[int, int] = {}

  def pre_step(self):
    self.agents.sort(key=lambda a: a.cur_opinion)
    for i, a in enumerate(self.agents):
      self.agent_indices[a.unique_id] = i
    self.epsilon = np.random.normal(0, self.noise_std, (self.num_nodes, ))

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    neighbor_ids = set([x.unique_id for x in neighbors + [agent]])
    o = agent.cur_opinion + self.epsilon[agent.unique_id]

    i_pre = self.agent_indices[agent.unique_id] - 1
    i_post = self.agent_indices[agent.unique_id] + 1
    ret: List[HKAgent] = []

    while len(ret) < count:
      no_pre = i_pre < 0
      no_post = i_post >= len(self.agents)
      if no_pre and no_post:
        break
      use_pre = no_post or (
          not no_pre and
          o - self.agents[i_pre].cur_opinion < self.agents[i_post].cur_opinion - o)
      if use_pre:
        a = self.agents[i_pre]
        if a.unique_id not in neighbor_ids:
          ret.append(a)
        i_pre -= 1
      else:
        a = self.agents[i_post]
        if a.unique_id not in neighbor_ids:
          ret.append(a)
        i_post += 1

    return ret


class OpinionRandom(HKModelRecommendationSystem):

  def __init__(
      self,
      model: HKModel,
      
      tolerance: float = 0.4,
      steepness: float = 1,
      noise_std: float = 0.1,
      random_ratio: float = 0,
  ):
    super().__init__(model)
    self.tolerance = tolerance
    self.steepness = steepness
    self.noise_std = noise_std
    self.random_ratio = random_ratio
    
    self.num_nodes = 0
    self.agents: List[HKAgent] = []
    self.all_indices = np.zeros((0, ), dtype=int)
    self.rate_mat = np.zeros((0, ), dtype=float)
    
  def post_init(self, dump_data: Optional[Any] = None):
    self.num_nodes = self.model.graph.number_of_nodes()
    self.all_indices = np.arange(self.num_nodes, dtype=int)
    # cache agents
    self.agents: List[HKAgent] = list(self.model.schedule.agents)
    if self.agents:
      assert self.agents[0].unique_id == 0
      assert self.agents[-1].unique_id == self.num_nodes - 1
    
  def pre_step(self):
    # calculate difference matrix
    opinion = np.array([x.cur_opinion for x in self.agents], dtype=float).reshape((1, -1))
    opinion_mat = np.repeat(opinion, self.num_nodes, axis=0)
    opinion_diff_mat = np.abs(opinion_mat - opinion_mat.T)
    
    # calculate rate matrix based on differences
    
    _i = np.diag_indices(self.num_nodes)
    
    raw_rate_mat = 1 - opinion_diff_mat / self.tolerance
    raw_rate_mat[raw_rate_mat < 0] = 0
    
    if self.noise_std > 0:
      noise_mat = np.random.normal(0, self.noise_std, raw_rate_mat.shape)
      raw_rate_mat = raw_rate_mat * (1 - 2 * noise_mat) + noise_mat
      raw_rate_mat[raw_rate_mat < 0] = 0
      
    np.fill_diagonal(raw_rate_mat, 0)
    
    if self.steepness != 1:
      raw_rate_mat = raw_rate_mat ** self.steepness
    
    # normalize rate matrix
    
    rate_sum_rev = np.reshape(np.sum(raw_rate_mat, axis=1), (-1, 1)) ** -1
    rate_mat = raw_rate_mat * rate_sum_rev
    if self.random_ratio > 0:
      rate_mat = (1 - self.random_ratio) * rate_mat + self.random_ratio / (self.num_nodes - 1)
    np.fill_diagonal(rate_mat, 0)
    
    # expose rate matrix
    self.rate_mat = rate_mat
  
    
  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    neighbor_ids = np.array([x.unique_id for x in neighbors + [agent]], dtype=int)
    raw_rate_vec = self.rate_mat[agent.unique_id]
    
    rate_vec = np.copy(raw_rate_vec)
    rate_vec[neighbor_ids] = 0
    rate_vec /= np.sum(rate_vec)
    
    candidates = np.random.choice(self.all_indices, (count,), replace=False, p=rate_vec)
    ret = [self.agents[c] for c in candidates]
    return ret
    
