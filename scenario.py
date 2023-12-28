from typing import Dict, Tuple, List, Optional, Any
from numpy.typing import NDArray

import dataclasses

import networkx as nx
import numpy as np

from model import HKModel, HKModelParams, HKAgent
from mesa import DataCollector

from tqdm import tqdm


@dataclasses.dataclass
class ScenarioParams:
  agent_count: int = 1000
  agent_follow: int = 20
  total_step: int = 1000
  data_interval: int = 1
  stat_interval: int = 20
  opinion_range: Tuple[int, int] = (-1, 1)

  def generate(self):
    graph: nx.DiGraph = nx.erdos_renyi_graph(
        n=self.agent_count,
        p=self.agent_follow / (self.agent_count - 1),
        directed=True
    )
    opinion = np.random.uniform(*self.opinion_range, (self.agent_count, ))
    return graph, opinion


StatsType = Tuple[Tuple[NDArray[Any], NDArray[Any]], int, float, float]


class Scenario:

  model: HKModel = None
  datacollector: DataCollector = None
  stats: Dict[int, StatsType] = None
  steps: int = 0

  def __init__(
      self,
      scenario_params: ScenarioParams,
      model_params: HKModelParams,
  ):
    self.scenario_params = scenario_params
    self.model_params = model_params
    self.stats = {}
    self.steps = 0

  def init_data(self, collect=True):
    self.datacollector = DataCollector(agent_reporters=dict(
        Opinion='cur_opinion',
        DiffNeighbor='diff_neighbor',
        DiffRecommended='diff_recommended'
    ))
    self.stats = {}
    if collect:
      self.datacollector.collect(self.model)
      self.add_stats()
    self.model.datacollector = self.datacollector

  def init(self):
    graph, opinion = self.scenario_params.generate()
    model = HKModel(graph, opinion, self.model_params)
    self.model = model
    self.steps = 0
    self.init_data()

  def dump(self):
    # graph
    graph = nx.DiGraph(self.model.graph)
    agents: List[HKAgent] = self.model.schedule.agents
    # opinion
    opinion = np.zeros((self.model.graph.number_of_nodes(), ), dtype=float)
    for a in agents:
      opinion[a.unique_id] = a.cur_opinion
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

    if self.steps % self.scenario_params.data_interval == 0:
      self.datacollector.collect(self.model)
    if self.steps % self.scenario_params.stat_interval == 0:
      self.add_stats()

  def step(self, count: int = 0):
    if count < 1:
      count = self.scenario_params.total_step
    for _ in tqdm(range(count)):
      self.step_once()

  def get_opinion_data(self):
    data = self.datacollector.get_agent_vars_dataframe()
    opinion = data['Opinion'].unstack().to_numpy()
    dn = data['DiffNeighbor'].unstack().to_numpy()
    dr = data['DiffRecommended'].unstack().to_numpy()
    return opinion, dn, dr

  def add_stats(self):
    self.stats[self.steps] = self.collect_stats()

  def collect_stats(self, hist_interval=0.05):
    digraph, opinion, _, _, _ = self.dump()
    graph = nx.Graph(digraph)
    n = graph.number_of_nodes()

    # distance distribution
    o_slice_mat = np.tile(opinion.reshape((opinion.size, 1)), opinion.size)
    o_sample = np.abs(o_slice_mat - o_slice_mat.T).flatten()
    distance_dist = np.histogram(
        o_sample, bins=np.arange(0, np.max(o_sample) + hist_interval, hist_interval))

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

    return distance_dist, triads_count, clustering, s_index
