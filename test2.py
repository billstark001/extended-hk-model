from typing import Callable, Mapping, cast, Optional, List, Tuple
from numpy.typing import NDArray

import os
import pickle
from matplotlib.axes import Axes
import networkx as nx
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

from ehk_model_old.base import HKModelParams, Scenario, SimulationParams, HKModel, HKModelRecommendationSystem
from ehk_model_old.env import RandomNetworkProvider, ScaleFreeNetworkProvider
from ehk_model_old.recsys import Random, Opinion, Structure, Mixed
import ehk_model_old.stats as stats

BASE_PATH = './run2'
os.makedirs(BASE_PATH, exist_ok=True)

# build scenarios

def stat_collectors_f(): return {
    # 'layout': stats.NetworkLayoutCollector(use_last=True, return_dict=True),
    'triads': stats.TriadsCountCollector(),
    'cluster': stats.ClusteringCollector(),
    's-index': stats.SegregationIndexCollector(),
    # 'in-degree': stats.InDegreeCollector(),
    'distance': stats.DistanceCollectorDiscrete(
      use_js_divergence=True, 
      use_debug_data=True, 
      hist_interval=0.08
    ),
}


def save_sim_result(S: Scenario, name: str):
  dump_data = S.dump()
  with open(os.path.join(BASE_PATH, name + '.pkl'), 'wb') as f:
    pickle.dump(dump_data, f)


network_provider = RandomNetworkProvider(
    agent_count=400,
    agent_follow=15,
)
sim_p_standard = SimulationParams(
    max_total_step=10000,
    model_stat_interval=20,
    opinion_change_error=1e-4,
    model_stat_collectors=stat_collectors_f()
)


# rewiring_rate_array = 10 ** np.arange(-2, 0 + 1/5, 1/4)
# decay_rate_array = 10 ** np.arange(-3, 0, 1/3)

rewiring_rate_array = np.array([0.01, 0.03, 0.05, 0.1, 0.3, 0.5])
decay_rate_array = np.array([0.005, 0.01, 0.05, 0.1, 0.5, 1])
n_sims = 5

n_gens = [
    # lambda m: Random(m),
    lambda m: Mixed(
        m,
        Random(m, 10),
        Opinion(m),
        0.1),
    lambda m: Mixed(
        m,
        Random(m, 10),
        Structure(m, noise_std=0.2, matrix_init=True),
        0.1),
]

n_gen_names = ['op', 'st']

# all parameters

        
def gen_params(r: float, d: float, g: float):
  return HKModelParams(
      tolerance=0.45,
      decay=d,
      rewiring_rate=r,
      recsys_count=10,
      recsys_factory=g
  )

do_step = False

if __name__ == '__main__':
  r = rewiring_rate_array[3]
  d = decay_rate_array[1]
  g = n_gens[0]
  params = gen_params(r, d, g)

  sim_p_standard.model_stat_collectors = stat_collectors_f()

  # simulate & collect
  
  _g = globals()
  S = cast(Scenario, _g['S'] if 'S' in _g else None)

  if do_step or S is None:
    S = Scenario(network_provider, params, sim_p_standard)
    S.init()
    S.iter()
    if S.steps not in S.stats:
      S.add_model_stats()
    
  S_data_steps, opinion, dn, dr, sum_n, sum_r, n_n, n_r = S.generate_agent_stats_v1()
  S_stats = S.generate_model_stats()
  S_stat_steps = S_stats['step']
  
  # n = 260
  # plt.plot(S_stats['distance-axis-o'][n], S_stats['distance-pmf-o'][n])
  # plt.plot(S_stats['distance-axis-o'][n], S_stats['distance-worst-o-pmf'][n])
  # plt.title(S_stat_steps[n])
  
  print(1)
  
  # collect indices
  
  n_edges = np.array(sorted(list(S.model.graph.out_degree), key=lambda x: x[0]))[:, 1]
  h_index = np.mean(n_n / n_edges[np.newaxis, :], axis=1)
  if h_index.shape[0] > 1:
    h_index[0] = h_index[1]
  s_index = S_stats['s-index']
  p_index = 1 - np.array(S_stats['distance-worst-o'])
  g_index = 1 - np.array(S_stats['distance-worst-s'])
  
  
  print(2)
  
  # calculate stats
  
  p_index_resmpl = interp1d(S_stat_steps, p_index, kind='linear')(S_data_steps)
  g_index_resmpl = interp1d(S_stat_steps, g_index, kind='linear')(S_data_steps)
  
  g_mask = g_index_resmpl <= np.max(g_index) * 0.95
  pat_diff = (h_index - p_index_resmpl)[g_mask]
  p_last = p_index[-1]
  pat_mean = np.mean(pat_diff)
  pat_std = np.std(pat_diff)
  
  print(3)
  
  # plot indices
  
  plt.plot(S_data_steps, h_index, linewidth=1)
  plt.plot(S_stat_steps, s_index, linewidth=1)
  plt.plot(S_stat_steps, p_index, linewidth=1)
  plt.plot(S_stat_steps, g_index, linewidth=1)
  plt.legend(['homophily', 'segregation', 'polarization', 'general'])
  
  print(4)
  
  plt.show()
