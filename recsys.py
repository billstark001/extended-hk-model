
from typing import List
import numpy as np
from model import HKAgent, HKModelRecommendationSystem


class Random(HKModelRecommendationSystem):
  
  num_nodes = 0
  
  def post_init(self):
    self.num_nodes = self.model.graph.number_of_nodes()
    self.all_ids = np.arange(1, self.num_nodes + 1, dtype=int)
  
  def pre_step(self):
    pass
  
  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], rate: float) -> List[HKAgent]:
    neighbor_ids = [x.unique_id for x in neighbors]
    available_ids = np.setdiff1d(self.all_ids, neighbor_ids)
    selected_ids = set(np.random.permutation(available_ids)[:int(rate * len(neighbors) + 0.5)])
    ret = [a for a in self.model.schedule.agents if a.unique_id in selected_ids]
    return ret
  
  def post_step(self):
    pass