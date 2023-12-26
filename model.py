from typing import List, Optional, Tuple, Callable, Union, Iterable, Dict
from numpy.typing import NDArray

import numpy as np
import networkx as nx
from mesa import Agent, DataCollector, Model
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
    self.cur_opinion = opinion if opinion is not None else np.random.uniform(-1, 1)

    # future state
    self.diff_neighbor = 0
    self.diff_recommended = 0
    self.next_opinion = self.cur_opinion
    self.next_follow: Optional[Tuple['HKAgent', 'HKAgent']] = None

  def step(self):
    # clear data
    self.next_opinion = self.cur_opinion
    self.diff_neighbor = 0
    self.diff_recommended = 0
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
    concordant_neighbor: List['HKAgent'] = []
    concordant_recommended: List['HKAgent'] = []
    discordant_recommended: List['HKAgent'] = []
    discordant_neighbor: List['HKAgent'] = []
    for a in neighbors:
      if abs(self.cur_opinion - a.cur_opinion) <= epsilon:
        concordant_neighbor.append(a)
      else:
        discordant_neighbor.append(a)
    for a in recommended:
      if abs(self.cur_opinion - a.cur_opinion) <= epsilon:
        concordant_recommended.append(a)
      else:
        discordant_recommended.append(a)

    # update value
    n_concordant = len(concordant_neighbor) + len(concordant_recommended)
    if n_concordant > 0:
      sum_n = sum(a.cur_opinion - self.cur_opinion for a in concordant_neighbor)
      sum_r = sum(a.cur_opinion - self.cur_opinion for a in concordant_recommended)
      self.diff_neighbor = sum_n / n_concordant
      self.diff_recommended = sum_r / n_concordant
      self.next_opinion += ((sum_r + sum_n) / n_concordant) * self.model.p.decay

    # handle rewiring
    if gamma > 0 and discordant_neighbor and concordant_recommended and np.random.uniform() < gamma:
      follow = np.random.choice(concordant_recommended)
      unfollow = np.random.choice(discordant_neighbor)
      self.next_follow = (unfollow, follow)


class HKModel(Model):

  def __init__(
      self,
      graph: nx.DiGraph,
      opinion: Union[None, Iterable[float], NDArray, Dict[int, float]],
      params: 'HKModelParams' = None,
      collect = False
  ):
    params = params if params is not None else HKModelParams()
    opinion = opinion if opinion is not None else \
      np.random.uniform(-1, 1, (graph.number_of_nodes(), ))
    self.graph = graph
    self.p = params
    self.recsys = params.recsys_factory(
        self) if params.recsys_factory else None
    self.collect = collect

    self.grid = NetworkGrid(self.graph)
    self.schedule = RandomActivation(self)
    for node in self.graph.nodes():
      a = HKAgent(node, self, opinion[node])
      self.grid.place_agent(a, node)
      self.schedule.add(a)
    if self.recsys:
      self.recsys.post_init()
      
    if self.collect:
      self.datacollector = DataCollector(agent_reporters=dict(Opinion='cur_opinion'))
      self.datacollector.collect(self)

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
    for a in agents:
      # opinion
      a.cur_opinion = a.next_opinion
      # rewiring
      if a.next_follow:
        unfollow, follow = a.next_follow
        self.graph.remove_edge(a.unique_id, unfollow.unique_id)
        self.graph.add_edge(a.unique_id, follow.unique_id)
        changed.extend([a.unique_id, unfollow.unique_id, follow.unique_id])
    if self.recsys:
      self.recsys.post_step(changed)
    # collect data
    if self.collect:
      self.datacollector.collect(self)

  def get_recommendation(self, agent: HKAgent, neighbors: Optional[List[HKAgent]] = None) -> List[HKAgent]:
    if not self.recsys:
      return []
    neighbors = neighbors if neighbors is not None else self.grid.get_neighbors(
        agent.unique_id, include_center=False)
    return self.recsys.recommend(agent, neighbors, self.p.recsys_count)


class HKModelRecommendationSystem(abc.ABC):

  def __init__(self, model: HKModel):
    self.model = model

  @abc.abstractmethod
  def post_init(self):
    pass

  @abc.abstractmethod
  def pre_step(self):
    pass

  @abc.abstractmethod
  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    pass

  @abc.abstractmethod
  def pre_commit(self):
    pass

  @abc.abstractmethod
  def post_step(self, changed: List[int]):
    pass


@dataclasses.dataclass
class HKModelParams:
  tolerance: float = 0.25
  decay: float = 1
  # retweet_rate: float = 0.3
  recsys_count: int = 10
  rewiring_rate: float = 0.1

  recsys_factory: Optional[Callable[[HKModel],
                                    HKModelRecommendationSystem]] = None
