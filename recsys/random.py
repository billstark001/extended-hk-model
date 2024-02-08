from typing import List, Dict, Optional, Any
import numpy as np
from base import HKAgent, HKModelRecommendationSystem


class Random(HKModelRecommendationSystem):

  num_nodes = 0
  agent_map: Dict[int, HKAgent] = {}

  def post_init(self, dump_data: Optional[Any] = None):
    self.num_nodes = self.model.graph.number_of_nodes()
    for a in self.model.schedule.agents:
      self.agent_map[a.unique_id] = a

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    exclude_ids = np.array([x.unique_id for x in neighbors])
    candidates = np.random.randint(
        0, self.num_nodes, (count + len(neighbors), ))
    ret = np.setdiff1d(candidates, exclude_ids)
    return [self.agent_map[a] for a in ret[:count]]
