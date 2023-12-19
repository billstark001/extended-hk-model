from typing import List, Dict
import numpy as np
from model import HKAgent, HKModelRecommendationSystem


class Random(HKModelRecommendationSystem):

  num_nodes = 0
  agent_map: Dict[int, HKAgent] = {}

  def post_init(self):
    self.num_nodes = self.model.graph.number_of_nodes()
    for a in self.model.schedule.agents:
      self.agent_map[a.unique_id] = a

  def pre_step(self):
    pass

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], rate: float) -> List[HKAgent]:
    exclude_ids = np.array([x.unique_id for x in neighbors])
    required_count = int(rate * len(neighbors) + 0.5)
    ret = np.zeros((0, ), dtype=int)
    for _ in range(10):
      candidates = np.random.randint(0, self.num_nodes, (required_count - ret.size, ))
      applied = np.setdiff1d(candidates, exclude_ids)
      exclude_ids = np.concatenate([exclude_ids, applied])
      ret = np.concatenate([ret, applied])
      if len(ret) >= required_count:
        ret = ret[:required_count]
        break
    return [self.agent_map[a] for a in ret]

  def post_step(self, changed: List[int]):
    pass
  
  def pre_commit(self):
    pass