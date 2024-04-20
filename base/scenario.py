from typing import Dict, Tuple, List, Optional, Any, Protocol, Union
from numpy.typing import NDArray

import dataclasses

import networkx as nx
import numpy as np

from base.model import HKModel, HKModelParams, HKAgent
from mesa import DataCollector

from tqdm import tqdm

from collections import Counter

from utils.stat import first_more_or_equal_than

StatsType = Dict[str, Union[NDArray, int, float]]


class EnvironmentProvider(Protocol):

  def generate(self, *args, **kwargs) -> Tuple[nx.DiGraph, NDArray]:
    pass


class StatCollector(Protocol):

  def collect(
      self,
      prefix: str,
      n: int,
      step: int,
      digraph: nx.DiGraph,
      graph: nx.Graph,
      opinion: NDArray,
  ) -> Union[float, NDArray, Dict[str, Union[float, NDArray]]]:
    pass


short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


@dataclasses.dataclass
class SimulationParams:
  max_total_step: int = 1000
  opinion_change_error: float = 1e-10
  halt_monitor_step: int = 60

  data_interval: int = 1
  stat_interval: Union[int, Dict[int, int]] = 20
  stat_collectors: Dict[str, StatCollector] = dataclasses.field(
      default_factory=dict)

  def __post_init__(self):
    self.stat_interval_index = np.array([] if not isinstance(
        self.stat_interval, dict) else sorted(self.stat_interval.keys()), dtype=int)
    self.last_steps = -1
    self.last_range = -1

  def needs_stat(self, steps: int):
    if isinstance(self.stat_interval, dict):
      _i = self.stat_interval
      if steps in _i:
        self.last_steps = -1
        self.last_range = -1
        return True

      if self.last_steps > 0 and self.last_range > 0 and steps == self.last_steps + 1:
        return steps % _i[self.last_range] == 0

      i_range = first_more_or_equal_than(self.stat_interval_index, steps)
      range = self.stat_interval_index[i_range] if i_range < self.stat_interval_index.size else self.stat_interval_index[-1]
      self.last_range = range
      self.last_steps = steps
      return steps % _i[range] == 0

    return steps % self.stat_interval == 0


class Scenario:

  model: HKModel = None
  datacollector: DataCollector = None
  stats: Dict[int, StatsType] = None
  steps: int = 0

  def __init__(
      self,
      env_provider: EnvironmentProvider,
      model_params: HKModelParams,
      sim_params: SimulationParams,
  ):
    self.env_provider = env_provider
    self.sim_params = sim_params
    self.stat_collectors = self.sim_params.stat_collectors
    self.model_params = model_params
    self.stats = {}
    self.steps = 0
    self.halt_monitor = [(0xffffff, 0xffffff)] * \
        self.sim_params.halt_monitor_step

  def init_data(self, collect=True):
    self.datacollector = DataCollector(agent_reporters=dict(
        Opinion='cur_opinion',
        DiffNeighbor='diff_neighbor',
        DiffRecommended='diff_recommended',
        SumNeighbor='sum_neighbor',
        SumRecommended='sum_recommended',
        NumNeighbor='n_neighbor',
        NumRecommended='n_recommended',
    ), model_reporters=dict(
        Step=lambda _: self.steps
    ))
    self.stats = {}
    if collect:
      self.add_data()
      self.add_stats()
    self.model.datacollector = self.datacollector
    self.halt_monitor = [(0xffffff, 0xffffff)] * \
        self.sim_params.halt_monitor_step

  def init(self, *args, **kwargs):
    graph, opinion = self.env_provider.generate(*args, **kwargs)
    model = HKModel(graph, opinion, self.model_params)
    self.model = model
    self.steps = 0
    self.init_data()

  def dump(self):
    # graph
    graph = nx.DiGraph(self.model.graph)
    for n in graph:
      del graph.nodes[n]['agent']

    # recsys
    model_dump = self.model.dump()

    # opinion
    opinion = self.get_current_opinion()

    # data
    c = self.datacollector
    data = (c.model_vars, c._agent_records, c.tables, self.halt_monitor)
    return graph, opinion, model_dump, data, self.stats, self.steps

  def load(
      self,
      graph: nx.DiGraph,
      opinion: Dict[int, float],
      model_dump: Optional[Any] = None,
      data: Optional[Tuple[dict, dict, dict, list]] = None,
      stats: Dict[int, StatsType] = None,
      step: int = 0,
  ):
    self.model = HKModel(
        graph, opinion, self.model_params, dump_data=model_dump)
    if data is not None:
      self.init_data(collect=False)
      v, r, t, m = data
      self.datacollector.model_vars = v
      self.datacollector._agent_records = r
      self.datacollector.tables = t
      self.halt_monitor = m
    else:
      self.init_data()
    if stats is not None:
      self.stats = stats
    self.steps = step or 0

  def step_once(self):
    c_edge, c_opinion = self.model.step()
    self.steps += 1

    # update monitor
    self.halt_monitor.pop(0)
    self.halt_monitor.append((c_edge, c_opinion))

    # stats
    if self.steps % self.sim_params.data_interval == 0:
      self.add_data()
    if self.sim_params.needs_stat(self.steps):
      self.add_stats()

  def step(self, count: int = 0):
    if count < 1:
      count = self.sim_params.max_total_step
    for _ in tqdm(range(count), bar_format=short_progress_bar):
      self.step_once()
      halt, _, __ = self.check_halt_cond()
      if halt:
        break

  def check_halt_cond(self):
    val1 = max(x[0] for x in self.halt_monitor)
    val2 = max(x[1] for x in self.halt_monitor)
    cond1 = val1 == 0
    cond2 = val2 < self.sim_params.opinion_change_error
    cond0 = self.steps >= self.sim_params.max_total_step
    ret = (cond1 and cond2) or cond0
    return ret, val1, val2

  def get_current_opinion(self):
    agents: List[HKAgent] = self.model.schedule.agents
    opinion = np.zeros((self.model.graph.number_of_nodes(), ), dtype=float)
    for a in agents:
      opinion[a.unique_id] = a.cur_opinion
    return opinion

  def get_opinion_data(self):
    data = self.datacollector.get_agent_vars_dataframe().unstack()
    steps = data.index.to_numpy()
    opinion = data['Opinion'].to_numpy()
    dn = data['DiffNeighbor'].to_numpy()
    dr = data['DiffRecommended'].to_numpy()
    sum_n = data['SumNeighbor'].to_numpy()
    sum_r = data['SumRecommended'].to_numpy()
    n_n = data['NumNeighbor'].to_numpy()
    n_r = data['NumRecommended'].to_numpy()
    return steps, opinion, dn, dr, sum_n, sum_r, n_n, n_r

  def add_data(self):
    self.datacollector.collect(self.model)

  def add_stats(self):
    self.stats[self.steps] = self.collect_stats()

  def collect_stats(self):

    digraph = self.model.graph
    graph = nx.Graph(digraph)
    n = graph.number_of_nodes()
    opinion = self.get_current_opinion()

    ret_dict = {}
    for stat_name in self.stat_collectors:
      collector = self.stat_collectors[stat_name]
      ret = collector.collect(
          prefix=stat_name,
          n=n,
          step=self.steps,
          digraph=digraph,
          graph=graph,
          opinion=opinion
      )
      if isinstance(ret, dict):
        ret_dict.update(ret)
      else:
        ret_dict[stat_name] = ret

    return ret_dict

  def generate_stats(self):
    step_indices = list(self.stats.keys())
    item_set = set()
    for v in self.stats.values():
      if isinstance(v, dict):
        for k in v.keys():
          item_set.add(k)

    ret_dict = {
        'step': step_indices
    }
    for item in item_set:
      ret_dict[item] = []

    for step in step_indices:
      step_dict = self.stats[step]
      if not isinstance(step_dict, dict):
        step_dict = {}
      for item in item_set:
        ret_dict[item].append(step_dict[item] if item in step_dict else None)

    return ret_dict
