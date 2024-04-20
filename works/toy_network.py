from typing import cast, Tuple, Any, List

import os

import networkx as nx
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
import seaborn as sns

from base import HKModelParams, Scenario, SimulationParams
from env import RandomNetworkProvider, ScaleFreeNetworkProvider
from recsys import Random, Opinion, Structure
import stats
from utils.stat import area_under_curve
from utils.plot import plt_figure

BASE_PATH = './fig_final'

os.makedirs(BASE_PATH, exist_ok=True)
mpl.rcParams['font.size'] = 18
sns.set_theme(style='whitegrid')

def plt_save_and_close(path: str, eps=True, png=True):
  if eps:
    plt.savefig(path + '.eps', dpi=300, bbox_inches='tight')
  if png:
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
  return plt.close()

stat_collectors = {
  'layout': stats.NetworkLayoutCollector(),
  's-index': stats.SegregationIndexCollector(),
  'distance': stats.DistanceCollectorDiscrete(
      use_js_divergence=True,
      hist_interval=0.08,
  ),
}

s_params = RandomNetworkProvider(
    agent_count=400,
    agent_follow=15,
)
sim_p_standard = SimulationParams(
    max_total_step=1000,
    stat_interval=15,
    stat_collectors=stat_collectors
)

params1= HKModelParams(
    tolerance=0.4,
    decay=0.05,
    rewiring_rate=0.01,
    recsys_count=10,
    recsys_factory=lambda m: Opinion(m),
)

params2 = HKModelParams(
    tolerance=0.4,
    decay=0.01,
    rewiring_rate=0.05,
    recsys_count=10,
    recsys_factory=lambda m: Opinion(m),
)

# colormap for network plotting

cmap = 'coolwarm'

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for i_p, params in enumerate((params1, params2)):
    
  S = Scenario(s_params, params, sim_p_standard)
  S.init()
  S.step()

  # generate stats

  if S.steps not in S.stats:
    S.add_stats()
    
  S_data_steps, opinion, dn, dr, sum_n, sum_r, n_n, n_r = S.get_opinion_data()
  S_stats = S.generate_stats()
  S_stat_steps = S_stats['step']
  
  fig, (axop, axst, axhp) = cast(Tuple[Any, List[Axes]], plt_figure(n_col=3, total_width=12))
  
  n_edges = np.array(sorted(list(S.model.graph.out_degree), key=lambda x: x[0]))[:, 1]
  h_index = np.mean(n_n / n_edges[np.newaxis, :], axis=1)
  if h_index.shape[0] > 1:
    h_index[0] = h_index[1]
  s_index = S_stats['s-index']
  p_index = 1 - np.array(S_stats['distance-worst-o'])
  g_index = 1 - np.array(S_stats['distance-worst-s'])
  
  # plot opinion data
  
  axop.plot(opinion, lw=0.5)
  axop.set_title('(a) opinion evolution')
  axop.set_xlabel('step')
  
  # plot indices
  
  axst.plot(S_data_steps, h_index, linewidth=1)
  axst.plot(S_stat_steps, s_index, linewidth=1)
  axst.plot(S_stat_steps, p_index, linewidth=1)
  axst.plot(S_stat_steps, g_index, linewidth=1)
  axst.legend(['homophily', 'segregation', 'polarization', 'general'])
  
  axst.set_title('(b) indices along simulation')
  axst.set_xlabel(f'step')
  axst.set_ylabel('index')
  
  # plot curves
  
  p_index_resmpl = interp1d(S_stat_steps, p_index, kind='linear')(S_data_steps)
  gradation = area_under_curve([p_index_resmpl, h_index])
  axhp.plot(p_index_resmpl, h_index)
  axhp.set_ylabel('homophily')
  axhp.set_xlabel('polarization')
  axhp.set_title(f'(c) H-P curves (AUC: {gradation:.4f})')


  plt_save_and_close(os.path.join(BASE_PATH, f'toy_stats_{i_p}'))
  
  # plot 3 snapshots

  fig, axes = cast(Tuple[Any, List[Axes]], plt_figure(n_col=3, hw_ratio=1, total_width=10))
  layout = S_stats['layout']
  for i, j in enumerate([0, 20, 40]):
    (pos, color, G) = layout[j]
    ax = axes[i]
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    nx.draw_networkx_nodes(
        G, ax=ax, pos=pos, node_color=color, cmap=cmap, vmin=-1, vmax=1, node_size=40)
    nx.draw_networkx_edges(G, ax=ax, pos=pos, node_size=40, alpha=0.36)
    ax.set_xlabel(f'step = {S_stat_steps[j]}')

  plt.colorbar(sm, ticks=np.linspace(-1, 1, 5), ax=axes[2])
  plt.subplots_adjust(wspace=0.1, hspace=0.1)

  plt.tight_layout()
  plt_save_and_close(os.path.join(BASE_PATH, f'toy_network_{i_p}'), eps=False)


