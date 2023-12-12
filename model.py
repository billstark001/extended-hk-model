from typing import List, Optional, Tuple, Callable

import random
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
import dataclasses
import abc


class HKAgent(Agent):

  model: 'HKModel' = None

  def __init__(
      self,
      unique_id: int,
      model: 'HKModel',
      opinion: Optional[float] = None
  ):
    super().__init__(unique_id, model)

    # current state
    self.cur_opinion = opinion if opinion is not None else random.uniform(0, 1)

    # future state
    self.next_opinion = 0
    self.next_follow: Optional[Tuple['HKAgent', 'HKAgent']] = None

  def step(self):
    # clear data
    self.next_opinion = self.cur_opinion
    self.next_follow = None

    # get the neighbors
    neighbors: List['HKAgent'] = self.model.grid.get_neighbors(
        self.unique_id, include_center=False)
    recommended: List['HKAgent'] = self.model.get_recommendation(
        self, neighbors)
    if not (len(neighbors) + len(recommended)):
      return

    # calculate concordant set
    epsilon = self.model.p.tolerance
    gamma = self.model.p.rewiring_rate
    concordant: List['HKAgent'] = []
    concordant_recommended: List['HKAgent'] = []
    discordant: List['HKAgent'] = []
    for a in neighbors:
      if abs(self.cur_opinion - a.cur_opinion) <= epsilon:
        concordant.append(a)
      else:
        discordant.append(a)
    for a in recommended:
      if abs(self.cur_opinion - a.cur_opinion) <= epsilon:
        concordant.append(a)
        concordant_recommended.append(a)
      else:
        discordant.append(a)

    # update value
    if concordant:
      avg_new_opinion = sum(
          [a.cur_opinion for a in concordant]) / len(concordant)
      self.next_opinion = avg_new_opinion

    # handle rewiring
    if gamma > 0 and discordant and concordant_recommended and random.random() < gamma:
      follow = random.choice(concordant_recommended)
      unfollow = random.choice(discordant)
      self.next_follow = (unfollow, follow)


class HKModel(Model):

  def __init__(
      self,
      graph: nx.DiGraph,
      params: 'HKModelParams',
  ):
    self.graph = graph
    self.p = params
    self.recsys = params.recsys_factory(
        self) if params.recsys_factory else None

    self.grid = NetworkGrid(self.graph)
    self.schedule = RandomActivation(self)
    for node in self.graph.nodes():
      a = HKAgent(node, self)
      self.grid.place_agent(a, node)
      self.schedule.add(a)
    if self.recsys:
      self.recsys.post_init()

  def step(self):
    agents: List['HKAgent'] = self.schedule.agents
    # let agents execute operations
    if self.recsys:
      self.recsys.pre_step()
    self.schedule.step()
    # commit changes
    for a in agents:
      a.cur_opinion = a.next_opinion
      if a.next_follow:
        unfollow, follow = a.next_follow
        self.graph.remove_edge(a.unique_id, unfollow.unique_id)
        self.graph.add_edge(a.unique_id, follow.unique_id)
    if self.recsys:
      self.recsys.post_step()

  def get_recommendation(self, agent: HKAgent, neighbors: Optional[List[HKAgent]] = None) -> List[HKAgent]:
    if not self.recsys:
      return []
    neighbors = neighbors if neighbors is not None else self.grid.get_neighbors(
        agent.unique_id, include_center=False)
    return self.recsys(agent, neighbors, self.p.recsys_rate)


class HKModelRecommendationSystem(abc.ABC):

  def __init__(self, model: HKModel):
    self.model = model

  @abc.abstractmethod
  def post_init():
    pass

  @abc.abstractmethod
  def pre_step():
    pass

  @abc.abstractmethod
  def recommend(agent: HKAgent, neighbors: List[HKAgent], rate: float) -> List[HKAgent]:
    pass

  @abc.abstractmethod
  def post_step():
    pass


@dataclasses.dataclass
class HKModelParams:
  tolerance: float = 0.25
  recsys_rate: float = 0.5
  rewiring_rate: float = 0.1

  recsys_factory: Optional[Callable[[HKModel],
                                    HKModelRecommendationSystem]] = None
