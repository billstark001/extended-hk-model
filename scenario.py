from typing import Dict, Tuple, List, Optional

import dataclasses

import networkx as nx
import numpy as np

from model import HKModel, HKModelParams, HKAgent
from mesa import DataCollector

from tqdm import tqdm


@dataclasses.dataclass
class ScenarioParams:
  n_agent: int = 1000
  n_edges: int = 20
  n_step: int = 1000
  opinion_range: Tuple[int, int] = (-1, 1)

  def generate(self):
    graph: nx.DiGraph = nx.erdos_renyi_graph(
        n=self.n_agent,
        p=self.n_edges / (self.n_agent - 1),
        directed=True
    )
    opinion = np.random.uniform(*self.opinion_range, (self.n_agent, ))
    return graph, opinion

class Scenario:

  model: HKModel = None
  datacollector: DataCollector = None

  def __init__(
      self,
      scenario_params: ScenarioParams,
      model_params: HKModelParams,
  ):
    self.scenario_params = scenario_params
    self.model_params = model_params
    
  def init_data(self, collect=True):
    self.datacollector = DataCollector(agent_reporters=dict(
      Opinion='cur_opinion',
      DiffNeighbor='diff_neighbor',
      DiffRecommended='diff_recommended'
    ))
    if collect:
      self.datacollector.collect(self.model)
    self.model.datacollector = self.datacollector

  def init(self):
    graph, opinion = self.scenario_params.generate()
    model = HKModel(graph, opinion, self.model_params)
    self.model = model
    self.init_data()

  def dump(self):
    # graph
    graph = nx.DiGraph(self.model.graph)
    agents: List[HKAgent] = []
    # opinion
    opinion = np.zeros((self.model.graph.number_of_nodes(), ), dtype=float)
    for a in agents:
      opinion[a.unique_id] = a.cur_opinion
    # data
    c = self.datacollector
    data = (c.model_vars, c._agent_records, c.tables)
    return graph, opinion, data

  def load(self, graph: nx.DiGraph, opinion: Dict[int, float], data: Optional[Tuple[dict, dict, dict]] = None):
    self.model = HKModel(graph, opinion, self.model_params)
    if data is not None:
      self.init_data(collect=False)
      v, r, t = data
      self.datacollector.model_vars = v
      self.datacollector._agent_records = r
      self.datacollector.tables = t
    else:
      self.init_data()
      
  def step_once(self):
    self.model.step()
    self.datacollector.collect(self.model)
    
  def step(self, count: int = 0):
    if count < 1:
      count = self.scenario_params.n_step
    for _ in tqdm(range(count)):
      self.step_once()
      
  def get_opinion_data(self):
    data = self.datacollector.get_agent_vars_dataframe()
    opinion = data['Opinion'].unstack().to_numpy()
    dn = data['DiffNeighbor'].unstack().to_numpy()
    dr = data['DiffRecommended'].unstack().to_numpy()
    return opinion, dn, dr
  
  def collect_stats(self):
    graph, o_slice, _ = self.dump()
    
    # distance distribution
    o_slice_mat = np.tile(o_slice.reshape((o_slice.size, 1)), o_slice.size)
    o_slice_dist = np.abs(o_slice_mat - o_slice_mat.T)
    
    # closed triads' count
    triads = nx.triangles(nx.Graph(graph))
    
    
    # clustering coefficient



    # segregation index
    
