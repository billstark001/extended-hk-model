from typing import Dict, Tuple, List, Optional, Any, Protocol, Union
from numpy.typing import NDArray

import dataclasses

import networkx as nx
import numpy as np

from base.model import HKModel, HKModelParams, HKAgent
from mesa import DataCollector

from tqdm import tqdm

from collections import Counter

StatsType = Dict[str, Union[NDArray, int, float]]

class EnvironmentProvider(Protocol):
  
  def generate(self, *args, **kwargs) -> Tuple[nx.DiGraph, NDArray]:
    pass
  
@dataclasses.dataclass
class SimulationParams:
  total_step: int = 1000
  data_interval: int = 1
  stat_interval: int = 20

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
    self.model_params = model_params
    self.stats = {}
    self.steps = 0

  def init_data(self, collect=True):
    self.datacollector = DataCollector(agent_reporters=dict(
        Opinion='cur_opinion',
        DiffNeighbor='diff_neighbor',
        DiffRecommended='diff_recommended'
    ), model_reporters=dict(
        Step=lambda _: self.steps
    ))
    self.stats = {}
    if collect:
      self.add_data()
      self.add_stats()
    self.model.datacollector = self.datacollector

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
      
    # opinion
    opinion = self.get_current_opinion()
    
    # data
    c = self.datacollector
    data = (c.model_vars, c._agent_records, c.tables)
    return graph, opinion, data, self.stats, self.steps

  def load(
      self,
      graph: nx.DiGraph,
      opinion: Dict[int, float],
      data: Optional[Tuple[dict, dict, dict]] = None,
      stats: Dict[int, StatsType] = None,
      step: int = 0,
  ):
    self.model = HKModel(graph, opinion, self.model_params)
    if data is not None:
      self.init_data(collect=False)
      v, r, t = data
      self.datacollector.model_vars = v
      self.datacollector._agent_records = r
      self.datacollector.tables = t
    else:
      self.init_data()
    if stats is not None:
      self.stats = stats
    self.steps = step or 0

  def step_once(self):
    self.model.step()
    self.steps += 1

    if self.steps % self.sim_params.data_interval == 0:
      self.add_data()
    if self.steps % self.sim_params.stat_interval == 0:
      self.add_stats()

  def step(self, count: int = 0):
    if count < 1:
      count = self.sim_params.total_step
    for _ in tqdm(range(count)):
      self.step_once()
      
  def should_halt(self):
    return self.steps >= self.sim_params.total_step

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
    return steps, opinion, dn, dr

  def add_data(self):
    self.datacollector.collect(self.model)

  def add_stats(self):
    self.stats[self.steps] = self.collect_stats()

  def collect_stats(self, hist_interval=0.05):
    digraph = self.model.graph
    graph = nx.Graph(digraph)
    n = graph.number_of_nodes()
    
    opinion = self.get_current_opinion()
    
    # in-degree distribution
    in_degree_dict = dict(digraph.in_degree())
    in_degree_dist = Counter(in_degree_dict.values())
    in_degree = np.array(sorted(in_degree_dist.items())).T

    # distance distribution
    o_slice_mat = np.tile(opinion.reshape((opinion.size, 1)), opinion.size)
    o_sample = np.abs(o_slice_mat - o_slice_mat.T).flatten()
    distance_dist = np.histogram(
        o_sample, bins=np.arange(0, np.max(o_sample) + hist_interval, hist_interval))

    # subjective distance distribution
    neighbors = np.array(digraph.edges)
    o_sample_neighbors = np.abs(opinion[neighbors[:, 1]] - opinion[neighbors[:, 0]])
    s_distance_dist = np.histogram(
        o_sample_neighbors, bins=np.arange(0, np.max(o_sample_neighbors) + hist_interval, hist_interval))

    # closed triads' count
    triads = nx.triangles(graph)
    triads_count = sum(triads.values()) // 3

    # clustering coefficient
    clustering = nx.average_clustering(graph)

    # segregation index
    positive_amount = max(1, np.sum(opinion > 0))
    negative_amount = max(1, n - positive_amount)
    edge_interconnection = len(
        [None for u, v in graph.edges if opinion[u] * opinion[v] <= 0])

    density = graph.number_of_edges() / (n * (n - 1) / 2)
    s_index: float = 1 - edge_interconnection / \
        (2 * density * positive_amount * negative_amount)
        
    ret_dict = {
      'in-degree': in_degree,
      'distance distribution': distance_dist,
      'subjective distance distribution': s_distance_dist,
      'closed triads\' count': triads_count,
      'clustering coefficient': clustering, 
      'segregation index': s_index,
    }

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
    
