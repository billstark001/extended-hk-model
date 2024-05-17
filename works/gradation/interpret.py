
import os

import pickle
import importlib

import numpy as np
from tqdm import tqdm

from base import Scenario

from base.scenario import SimulationParams
import works.gradation.simulate as p

import stats

import utils.plot as _p
importlib.reload(_p)


# parameters

scenario_base_path = './run2_old'
scenario_base_path_2 = './run2'
plot_path = './fig2'

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(scenario_base_path_2, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)



# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


sim_p_standard = SimulationParams(
    max_total_step=15000,
    model_stat_interval={
        50: 5,
        150: 10,
        300: 15,
        114514: 20,
    },
    opinion_change_error=1e-4,
    model_stat_collectors={
      'triads': stats.TriadsCountCollector(),
      'cluster': stats.ClusteringCollector(),
      's-index': stats.SegregationIndexCollector(),
      'in-degree': stats.InDegreeCollector(),
      'distance': stats.DistanceCollectorDiscrete(
          use_js_divergence=True,
          hist_interval=0.08,
      ),
      'layout': stats.NetworkLayoutCollector(use_last=True)
    },
    agent_stat_keys=[
      'cur_opinion',
      'n_neighbor', 'n_recommended',
      'diff_neighbor', 'diff_recommended',
      'sum_neighbor', 'sum_recommended',
      'has_follow_event', 'unfollowed', 'followed',
  ]
)


if __name__ == '__main__':

  for scenario_name, r, d, g in tqdm(p.params_arr, bar_format=short_progress_bar):

    # load scenario

    params = p.gen_params(r, d, g)
    S = Scenario(p.network_provider, params, sim_p_standard)

    scenario_path = os.path.join(scenario_base_path, scenario_name + '.pkl')
    scenario_path_2 = os.path.join(scenario_base_path_2, scenario_name + '_record.pkl')
    if not os.path.exists(scenario_path) or os.path.exists(scenario_path_2):
      continue

    with open(scenario_path, 'rb') as f:
      dump_data = pickle.load(f)
    S.load(*dump_data)

    # collect stats
    
    metadata, model_stats, agent_stats = S.generate_record_data()
    new_agent_stats = {
      'step': agent_stats['step'],
      'cur_opinion': agent_stats['cur_opinion'],
    }
    
    n1 = agent_stats['n_neighbor'][..., np.newaxis]
    n2 = agent_stats['n_recommended'][..., np.newaxis]
    n3 = np.zeros_like(n1)
    n4 = np.zeros_like(n2)
    
    new_agent_stats['nr_agents'] = np.concatenate([n1, n2, n3, n4], axis=-1)
    
    n1 = agent_stats['sum_neighbor'][..., np.newaxis]
    n2 = agent_stats['sum_recommended'][..., np.newaxis]
    n3 = np.zeros_like(n1)
    n4 = np.zeros_like(n2)
    
    new_agent_stats['op_sum_agents'] = np.concatenate([n1, n2, n3, n4], axis=-1)
    
    f1 = agent_stats['has_follow_event'][..., np.newaxis].astype(int)
    f2 = agent_stats['unfollowed'][..., np.newaxis]
    f3 = agent_stats['followed'][..., np.newaxis]
    
    new_agent_stats['follow_event'] = np.concatenate([f1, f2, f3], axis=-1)
    
      
    if 'layout' in model_stats:
      layout = model_stats['layout']
      del model_stats['layout']
      o = []
      g = []
      for slice in layout:
        _, opinion, graph = slice 
        o.append(opinion)
        g.append(graph)
      model_stats['layout-opinion'] = o
      model_stats['layout-graph'] = g

    with open(scenario_path_2, 'wb') as f:
      pickle.dump((metadata, model_stats, new_agent_stats), f)
      
    print(scenario_name)
