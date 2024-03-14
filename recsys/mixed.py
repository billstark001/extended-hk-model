from typing import List, Dict, Optional, Any
import numpy as np
from base import HKAgent, HKModel, HKModelRecommendationSystem


class Mixed(HKModelRecommendationSystem):
  
  def __init__(
    self, 
    model: HKModel, 
    model1: HKModelRecommendationSystem, 
    model2: HKModelRecommendationSystem, 
    model1_ratio: float = 0.5
  ):
    super().__init__(model=model)
    self.model1 = model1
    self.model2 = model2
    self.r = model1_ratio
    
  def dump(self) -> Any:
    return self.model1.dump(), self.model2.dump()
  
  def post_init(self, dump_data: Optional[Any] = None):
    if dump_data is None:
      return
    d1, d2 = dump_data
    self.model1.post_init(dump_data=d1)
    self.model2.post_init(dump_data=d2)

  def pre_step(self):
    self.model1.pre_step()
    self.model2.pre_step()

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    count1 = max(int(count * self.r + 0.5), 0)
    count2 = max(count - count1, 0)
    l1 = self.model1.recommend(agent, neighbors, count1)
    l2 = self.model2.recommend(agent, neighbors, count2)
    ans = {}
    for a in l1 + l2:
      ans[a.unique_id] = a
    return list(ans.values())

  def pre_commit(self):
    self.model1.pre_commit()
    self.model2.pre_commit()

  def post_step(self, changed: List[int]):
    self.model1.post_step(changed)
    self.model2.post_step(changed)
  