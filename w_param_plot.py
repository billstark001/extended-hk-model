from typing import Callable, Mapping, cast, Optional, List, Tuple
from numpy.typing import NDArray

import os
import pickle
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

from base import HKModelParams, Scenario, SimulationParams
from base.model import HKModel, HKModelRecommendationSystem
from env import RandomNetworkProvider, ScaleFreeNetworkProvider
from recsys import Random, Opinion, Structure
import stats

import w_param_search as p

from w_logger import logger

from tqdm import tqdm

scenario_base_path = './run2'
plot_path = './fig2'

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)


def plot_network_snapshot(
    pos: Mapping,
    opinion: NDArray,
    G: nx.Graph,
    ax: Optional[Axes] = None,
    step: int = 0,
    cmap: str = 'coolwarm'
):
  norm = mpl.colors.Normalize(vmin=-1, vmax=1)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])

  mpl.rcParams['font.size'] = 18

  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['right'].set_visible(False)

  nx.draw_networkx_nodes(G, ax=ax, pos=pos, node_color=opinion,
                         cmap=cmap, vmin=-1, vmax=1, node_size=40)
  nx.draw_networkx_edges(G, ax=ax, pos=pos, node_size=40, alpha=0.36)

  ax.set_xlabel(f't = {step}')

  plt.colorbar(sm, ticks=np.linspace(-1, 1, 5), ax=ax)
  plt.tight_layout()


def plt_figure(n_row=1, n_col=1, hw_ratio=3/4, total_width=16, **kwargs) -> Tuple[Figure, List[Axes]]:
  width = total_width / n_col
  height = width * hw_ratio
  total_height = height * n_row
  return plt.subplots(n_row, n_col, figsize=(total_width, total_height), **kwargs)


# build scenario


for scenario_name, r, d, g in tqdm(p.params_arr):

  # load scenario

  params = p.gen_params(r, d, g)
  p.sim_p_standard.stat_collectors = p.stat_collectors_f()
  S = Scenario(p.network_provider, params, p.sim_p_standard)

  scenario_path = os.path.join(scenario_base_path, scenario_name + '.pkl')
  if not os.path.exists(scenario_path):
    continue

  with open(scenario_path, 'rb') as f:
    dump_data = pickle.load(f)
  S.load(*dump_data)
  
  # collect stats
  
  if S.steps not in S.stats:
    S.add_stats()
    
  S_data_steps, opinion, dn, dr, sum_n, sum_r, n_n, n_r = S.get_opinion_data()
  S_stats = S.generate_stats()
  S_stat_steps = S_stats['step']
  
  # collect indices
  
  h_index = np.mean(n_n, axis=1) / p.network_provider.agent_follow
  s_index = S_stats['s-index']
  p_index = 1 - np.array(S_stats['distance-worst-o'])
  g_index = 1 - np.array(S_stats['distance-worst-s'])
  
  # plot indices
  
  plt.plot(S_data_steps, h_index, linewidth=1)
  plt.plot(S_stat_steps, s_index, linewidth=1)
  plt.plot(S_stat_steps, p_index, linewidth=1)
  plt.plot(S_stat_steps, g_index, linewidth=1)
  plt.legend(['homophily', 'segregation', 'polarization', 'general'])
  
  plt.title(scenario_name)
  plt.xlabel(f'step (total: {S.steps})')
  
  plt.savefig(os.path.join(plot_path, scenario_name + '_stats.png'), bbox_inches='tight')
  plt.close()
    
  # plot networks
  
  layouts = S_stats['layout']
    
  _, (r1, r2) = cast(List[List[Axes]], plt_figure(n_col=4, n_row=2))
  all_axes = list(r1) + list(r2)
  all_indices = np.array(np.linspace(0, len(layouts) - 1, len(all_axes)), dtype=int)
  
  
  plotted_indices = set()
  for i_ax, i in enumerate(all_indices):
    if i in plotted_indices:
      continue
    step = S_stat_steps[i]
    pos, opinion, graph = layouts[i]
    plot_network_snapshot(pos, opinion, graph, all_axes[i_ax], step)
    plotted_indices.add(i)
    
  plt.savefig(os.path.join(plot_path, scenario_name + '_snapshot.png'), bbox_inches='tight')
  plt.close()

  print(scenario_name)
