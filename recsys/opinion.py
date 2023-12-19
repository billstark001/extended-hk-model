from typing import List, Dict
import numpy as np
from model import HKAgent, HKModel, HKModelRecommendationSystem


class Opinion(HKModelRecommendationSystem):

  num_nodes = 0
  epsilon = np.zeros((0, ))

  def __init__(self, model: HKModel, sigma: float = 0.03):
    super().__init__(model)
    self.sigma = sigma

  def post_init(self):
    self.num_nodes = self.model.graph.number_of_nodes()
    self.agents: List[HKAgent] = list(self.model.schedule.agents)
    self.agent_indices: Dict[int, int] = {}

  def pre_step(self):
    self.agents.sort(key=lambda a: a.cur_opinion)
    for i, a in enumerate(self.agents):
      self.agent_indices[a.unique_id] = i
    self.epsilon = np.random.normal(0, self.sigma, (self.num_nodes, ))

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], rate: float) -> List[HKAgent]:
    neighbor_ids = set([x.unique_id for x in neighbors])
    count = int(rate * len(neighbors) + 0.5)
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

  def post_step(self, changed: List[int]):
    pass

  def pre_commit(self):
    pass
