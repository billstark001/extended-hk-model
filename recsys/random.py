from typing import List, Dict
import numpy as np
from model import HKAgent, HKModelRecommendationSystem


class Random(HKModelRecommendationSystem):

  num_nodes = 0

  def post_init(self):
    self.num_nodes = self.model.graph.number_of_nodes()

  def pre_step(self):
    pass

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], rate: float) -> List[HKAgent]:
    exclude_ids = np.array([x.unique_id for x in neighbors])
    required_count = int(rate * len(neighbors) + 0.5)
    ret = np.zeros((0, ))
    for _ in range(10):
      candidates = np.random.randint(1, self.num_nodes + 1, (required_count - ret, ))
      applied = np.setdiff1d(candidates, exclude_ids)
      exclude_ids = np.concatenate(exclude_ids, applied)
      ret = np.concatenate(ret, applied)
      if len(ret) >= required_count:
        ret = ret[:required_count]
        break
    return ret.tolist()

  def post_step(self):
    pass