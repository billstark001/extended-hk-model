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

_sim_data_keys = [
    'cur_opinion',
    'n_neighbor', 'n_recommended',
    'diff_neighbor', 'diff_recommended',
    'sum_neighbor', 'sum_recommended',
    'has_follow_event', 'unfollowed', 'followed',
]


@dataclasses.dataclass
class SimulationParams:
  max_total_step: int = 1000
  opinion_change_error: float = 1e-10
  halt_monitor_step: int = 60

  agent_stat_interval: int = 1
  model_stat_interval: Union[int, Dict[int, int]] = 20
  model_stat_collectors: Dict[str, StatCollector] = dataclasses.field(
      default_factory=dict)

  def __post_init__(self):
    self.model_stat_interval_index = np.array([] if not isinstance(
        self.model_stat_interval, dict) \
          else sorted(self.model_stat_interval.keys()), dtype=int)
    self.last_steps = -1
    self.last_range = -1

  def needs_stat(self, steps: int):
    if isinstance(self.model_stat_interval, dict):
      _i = self.model_stat_interval
      if steps in _i:
        self.last_steps = -1
        self.last_range = -1
        return True

      if self.last_steps > 0 and self.last_range > 0 and steps == self.last_steps + 1:
        return steps % _i[self.last_range] == 0

      i_range = first_more_or_equal_than(self.model_stat_interval_index, steps)
      range = self.model_stat_interval_index[i_range] if i_range < self.model_stat_interval_index.size else self.model_stat_interval_index[-1]
      self.last_range = range
      self.last_steps = steps
      return steps % _i[range] == 0

    return steps % self.model_stat_interval == 0


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
    self.stat_collectors = self.sim_params.model_stat_collectors
    self.model_params = model_params
    self.stats = {}
    self.steps = 0
    self.halt_monitor = [(0xffffff, 0xffffff)] * \
        self.sim_params.halt_monitor_step

  def init_data(self, collect=True):
    self.datacollector = DataCollector(
        agent_reporters=dict((x, x) for x in _sim_data_keys),
        model_reporters=dict(
            step=lambda _: self.steps
        ))
    self.stats = {}
    if collect:
      self.add_agent_stats()
      self.add_model_stats()
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

  def iter_one_step(self):
    c_edge, c_opinion = self.model.step()
    self.steps += 1

    # update monitor
    self.halt_monitor.pop(0)
    self.halt_monitor.append((c_edge, c_opinion))

    # stats
    if self.steps % self.sim_params.agent_stat_interval == 0:
      self.add_agent_stats()
    if self.sim_params.needs_stat(self.steps):
      self.add_model_stats()

  def iter(self, count: int = 0):
    if count < 1:
      count = self.sim_params.max_total_step
    for _ in tqdm(range(count), bar_format=short_progress_bar):
      self.iter_one_step()
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

  def add_agent_stats(self):
    self.datacollector.collect(self.model)

  def add_model_stats(self):
    self.stats[self.steps] = self.collect_model_stats()

  def collect_model_stats(self):

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

  def generate_agent_stats(self):
    data = self.datacollector.get_agent_vars_dataframe().unstack()
    results = {
        'step': self.datacollector.get_model_vars_dataframe()['step'].to_numpy(),
    }
    for k in _sim_data_keys:
      results[k] = data[k].to_numpy()
    return results

  def generate_agent_stats_v1(self):
    results = self.generate_agent_stats()
    return (
        results['step'],
        results['cur_opinion'],
        results['diff_neighbor'],
        results['diff_recommended'],
        results['sum_neighbor'],
        results['sum_recommended'],
        results['n_neighbor'],
        results['n_recommended'],
    )

  def generate_model_stats(self):
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
  
  def generate_record_data(self):
    
    # record last step model stats
    if self.steps not in self.stats:
      self.add_model_stats()
      
    model_stats = self.generate_model_stats()
    agent_stats = self.generate_agent_stats()
    
    n_edges_ = np.array(
        sorted(list(self.model.graph.out_degree), key=lambda x: x[0]))
    node_indices = n_edges_[:, 0]
    n_edges = n_edges_[:, 1]
    
    metadata = dict(
      total_steps=self.steps,
      model_params = self.model.p,
      node_indices = node_indices,
      n_edges = n_edges,
    )
    
    return metadata, model_stats, agent_stats
