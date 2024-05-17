from typing import List, Optional, Tuple, Callable, Union, Iterable, Dict, Any, Set
from numpy.typing import NDArray

import numpy as np
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
    self.cur_opinion = opinion if opinion is not None else np.random.uniform(-1, 1)

    # future state
    self.next_opinion = self.cur_opinion
    self.next_follow: Optional[Tuple['HKAgent', 'HKAgent']] = None
    
    # recorded data
    self.nr_agents = [0, 0, 0, 0]
    self.op_sum_agents = [0, 0, 0, 0]
    self.follow_event = [False, -1, -1]

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
    
    concordant_neighbor: List['HKAgent'] = []
    concordant_recommended: List['HKAgent'] = []
    discordant_neighbor: List['HKAgent'] = []
    discordant_recommended: List['HKAgent'] = []
    
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
    n_neighbor = len(concordant_neighbor)
    n_recommended = len(concordant_recommended)
    n_concordant = n_neighbor + n_recommended
    
    if 'nr_agents' in self.model.collect:
      self.nr_agents = [n_neighbor, n_recommended, len(discordant_neighbor), len(discordant_recommended)]
    
    sum_n = 0
    sum_r = 0
    if n_concordant > 0:
      sum_n = sum(a.cur_opinion - self.cur_opinion for a in concordant_neighbor)
      sum_r = sum(a.cur_opinion - self.cur_opinion for a in concordant_recommended)
      
      self.next_opinion += ((sum_r + sum_n) / n_concordant) * self.model.p.decay
      
    if 'op_sum_agents' in self.model.collect:
      self.op_sum_agents = [
        sum_n, 
        sum_r, 
        sum(a.cur_opinion - self.cur_opinion for a in discordant_neighbor),
        sum(a.cur_opinion - self.cur_opinion for a in discordant_recommended)
      ] 

    # handle rewiring
    if gamma > 0 and discordant_neighbor and concordant_recommended and np.random.uniform() < gamma:
      follow = np.random.choice(concordant_recommended)
      unfollow = np.random.choice(discordant_neighbor)
      self.next_follow = (unfollow, follow)
      
    if 'follow_event' in self.model.collect:
      self.follow_event = [True, self.next_follow[0].unique_id, self.next_follow[1].unique_id] \
        if self.next_follow is not None else [False, -1, -1]


class HKModel(Model):

  def __init__(
      self,
      graph: nx.DiGraph,
      opinion: Union[None, Iterable[float], NDArray, Dict[int, float]],
      params: 'HKModelParams' = None,
      collect: Optional[Set[str]] = None,
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


@dataclasses.dataclass
class HKModelParams:
  tolerance: float = 0.25
  decay: float = 1
  # retweet_rate: float = 0.3
  recsys_count: int = 10
  rewiring_rate: float = 0.1

  recsys_factory: Optional[Callable[[HKModel],
                                    HKModelRecommendationSystem]] = None
  
  def to_dict(self) -> Dict[str, Any]:
    ret = dataclasses.asdict(self)
    del ret['recsys_factory']
    return ret
