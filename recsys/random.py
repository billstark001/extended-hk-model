from typing import List, Dict, Optional, Any
import numpy as np
from base import HKAgent, HKModelRecommendationSystem
from base.model import HKModel


class Random(HKModelRecommendationSystem):

  
  def __init__(self, model: HKModel, rec_count = 30):
    super().__init__(model)
    self.rec_count = rec_count
    self.num_nodes = 0
    self.agent_map: Dict[int, HKAgent] = {}
    self.candidates = np.zeros((0, 0))

  def post_init(self, dump_data: Optional[Any] = None):
    self.num_nodes = self.model.graph.number_of_nodes()
    for a in self.model.schedule.agents:
      self.agent_map[a.unique_id] = a
      
  def pre_step(self):
    self.candidates = np.random.randint(0, self.num_nodes, (self.num_nodes, self.rec_count))

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    exclude_ids = np.array([x.unique_id for x in neighbors + [agent]])
    candidates = self.candidates[agent.unique_id]
    ret = np.setdiff1d(candidates, exclude_ids)
    return [self.agent_map[a] for a in ret[:count]]
