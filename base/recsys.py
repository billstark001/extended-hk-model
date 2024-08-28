from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Any

import abc

if TYPE_CHECKING:
  from base.model import HKModel
  from base.agent import HKAgent

class HKModelRecommendationSystem(abc.ABC):

  def __init__(self, model: HKModel):
    self.model = model

  def post_init(self, dump_data: Optional[Any] = None):
    pass

  def pre_step(self):
    pass

  @abc.abstractmethod
  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    pass

  def pre_commit(self):
    pass

  def post_step(self, changed: List[int]):
    pass
  
  def dump(self) -> Any:
    return None
