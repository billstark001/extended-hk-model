from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Callable, Union, Iterable, Dict, Any, Set
from numpy.typing import NDArray

import numpy as np
import networkx as nx
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
import dataclasses

from base.agent import HKAgent

if TYPE_CHECKING:
  from base.recsys import HKModelRecommendationSystem


class HKModel(Model):

  def __init__(
      self,
      graph: nx.DiGraph,
      opinion: Union[None, Iterable[float], NDArray, Dict[int, float]],
      params: 'HKModelParams' = None,
      collect: Optional[Set[str]] = None,
      event_logger: Optional[Callable[[Any], None]] = None,
      dump_data: Optional[Any] = None,
  ):
    super().__init__()

    params = params if params is not None else HKModelParams()
    opinion = opinion if opinion is not None else \
        np.random.uniform(-1, 1, (graph.number_of_nodes(), ))
    self.graph = graph
    self.p = params
    self.recsys = params.recsys_factory(
        self) if params.recsys_factory else None
    self.collect = collect or set()

    self.event_logger = event_logger

    self.grid = NetworkGrid(self.graph)
    self.schedule = RandomActivation(self)
    for node in self.graph.nodes():
      a = HKAgent(node, self, opinion[node])
      self.grid.place_agent(a, node)
      self.schedule.add(a)
    if self.recsys:
      self.recsys.post_init(dump_data)

  def dump(self):
    return self.recsys.dump()

  def step(self):
    agents: List['HKAgent'] = self.schedule.agents
    # let agents execute operations
    if self.recsys:
      self.recsys.pre_step()
    self.schedule.step()
    # commit changes
    if self.recsys:
      self.recsys.pre_commit()

    changed: List[int] = []
    changed_count = 0
    changed_opinion_max = 0.

    for a in agents:
      # opinion
      changed_opinion = a.next_opinion - a.cur_opinion
      a.cur_opinion = a.next_opinion
      changed_opinion_max = max(changed_opinion_max, abs(changed_opinion))
      # rewiring
      if a.next_follow:
        unfollow, follow = a.next_follow
        self.graph.remove_edge(a.unique_id, unfollow.unique_id)
        self.graph.add_edge(a.unique_id, follow.unique_id)
        changed.extend([a.unique_id, unfollow.unique_id, follow.unique_id])
        changed_count += 1

    if self.recsys:
      self.recsys.post_step(changed)

    return changed_count, changed_opinion_max

  def get_recommendation(self, agent: HKAgent, neighbors: Optional[List[HKAgent]] = None) -> List[HKAgent]:
    if not self.recsys:
      return []
    neighbors = neighbors if neighbors is not None else self.grid.get_neighbors(
        agent.unique_id, include_center=False)
    return self.recsys.recommend(agent, neighbors, self.p.recsys_count)


@dataclasses.dataclass
class HKModelParams:
  # epsilon
  tolerance: float = 0.25

  decay: float = 1
  rewiring_rate: float = 0.1
  retweet_rate: float = 0.3

  recsys_count: int = 10
  recsys_factory: Optional[
      Callable[[HKModel], HKModelRecommendationSystem]] = None
  
  tweet_retain_count: int = 3

  def to_dict(self) -> Dict[str, Any]:
    ret = dataclasses.asdict(self)
    del ret['recsys_factory']
    return ret
