from typing import cast, List, Tuple, Any, Optional
from numpy.typing import NDArray

import os
import json
import pickle
import importlib
import dataclasses

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes

from scipy.interpolate import interp1d

from base import Scenario

import w_param_search as p
import w_plot_utils as _p
importlib.reload(_p)

from w_plot_utils import plot_network_snapshot, plt_figure

# parameters

scenario_base_path = './run2'
plot_path = './fig2'

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

pat_csv_path = os.path.join(plot_path, 'pattern_stats.json')

do_plot = False
do_plot_layout = False

@dataclasses.dataclass
class ScenarioPatternRecorder:
  name: str
  step: int
  active_step: int
  
  p_last: float
  s_last: float
  h_last: float
  
  pat_abs_mean: float
  pat_abs_std: float
  
  pat_area_hp: float
  pat_area_ps: float
  
  cluster: float
  triads: float
  in_degree: Tuple[float, float, float]
  opinion_diff: float
  
  
import numpy as np

def area_under_curve(points: NDArray, arg_sort = False, complement: Optional[float]=1):
  # points: (n, 2)
  
  x = points[0]
  y = points[1]
  if arg_sort:
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]
  
  dx = np.diff(x)
  area = 0.5 * np.sum(dx * (y[1:] + y[:-1]))
  if complement:
    area += (complement - x[-1]) * y[-1]
  
  return area
  

pat_columns = ['name', 'step', 'active_step', 'pat_mean', 'pat_std', 'h_last']

active_threshold = 0.98

# utilities

mpl.rcParams['font.size'] = 18


# build scenario

if __name__ == '__main__':

  pat_stats_set: List[ScenarioPatternRecorder] = []

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
    
    n_edges = np.array(sorted(list(S.model.graph.out_degree), key=lambda x: x[0]))[:, 1]
    h_index = np.mean(n_n / n_edges[np.newaxis, :], axis=1)
    if h_index.shape[0] > 1:
      h_index[0] = h_index[1]
    s_index = S_stats['s-index']
    p_index = 1 - np.array(S_stats['distance-worst-o'])
    g_index = 1 - np.array(S_stats['distance-worst-s'])
    
    # calculate stats
    
    p_index_resmpl = interp1d(S_stat_steps, p_index, kind='linear')(S_data_steps)
    g_index_resmpl = interp1d(S_stat_steps, g_index, kind='linear')(S_data_steps)
    
    g_mask = g_index_resmpl <= np.max(g_index) * active_threshold
    pat_diff = (h_index - p_index_resmpl)[g_mask]
    
    opinion_last = opinion[-1]
    opinion_last_mean = np.mean(opinion_last)
    opinion_last_diff = \
      np.mean(opinion_last[opinion_last > opinion_last_mean]) - \
        np.mean(opinion_last[opinion_last <= opinion_last_mean])
    
    pat_stats = ScenarioPatternRecorder(
      name = scenario_name,
      step = S.steps,
      active_step = int(np.sum(g_mask, dtype=int)),
      p_last = p_index[-1],
      h_last = h_index[-1],
      s_last = s_index[-1],
      pat_abs_mean = np.mean(pat_diff),
      pat_abs_std = np.std(pat_diff),
      pat_area_hp = area_under_curve([p_index_resmpl, h_index]),
      pat_area_ps = area_under_curve([s_index, p_index]),
      cluster = S_stats['cluster'][-1],
      triads = S_stats['triads'][-1],
      in_degree = [S_stats[x][-1] for x in ['in-degree-alpha', 'in-degree-p-value', 'in-degree-R']],
      opinion_diff = opinion_last_diff if np.isfinite(opinion_last_diff) else -1,
    )
    
    pat_stats_set.append(dataclasses.asdict(pat_stats))
    with open(pat_csv_path, 'w', encoding='utf8') as f:
      json.dump(pat_stats_set, f, indent=2, ensure_ascii=False)

    if not do_plot:
      continue
    # plot indices
    fig, (ax, axhp, axps) = cast(Tuple[Any, List[Axes]], plt_figure(n_col=3))
    
    ax.plot(S_data_steps, h_index, linewidth=1)
    ax.plot(S_stat_steps, s_index, linewidth=1)
    ax.plot(S_stat_steps, p_index, linewidth=1)
    ax.plot(S_stat_steps, g_index, linewidth=1)
    ax.legend(['homophily', 'segregation', 'polarization', 'general'])
    
    ax.set_title(scenario_name)
    ax.set_xlabel(f'step (total: {S.steps})')
    
    # plot curves
    
    axhp.plot(p_index_resmpl, h_index)
    axhp.set_ylabel('homophily')
    axhp.set_xlabel('polarization')
    axhp.set_title(pat_stats.pat_area_hp)
    
    axps.plot(s_index, p_index)
    axps.set_ylabel('polarization')
    axps.set_xlabel('segregation')
    axps.set_title(pat_stats.pat_area_ps)
    
    plt.savefig(os.path.join(plot_path, scenario_name + '_stats.png'), bbox_inches='tight')
    plt.close()
    
      
    # plot networks
    
    if not do_plot_layout:
      continue
    
    layouts = S_stats['layout']
      
    _, (r1, r2) = cast(Tuple[Any, List[List[Axes]]], plt_figure(n_col=4, n_row=2))
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
