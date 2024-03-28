from typing import cast, List

import os
import pickle
import importlib

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes

from scipy.interpolate import interp1d

from base import Scenario

import w_param_search as p
from w_plot_utils import plot_network_snapshot, plt_figure

import w_plot_utils as _p
importlib.reload(_p)

# parameters

scenario_base_path = './run2'
plot_path = './fig2'

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

pat_csv_path = os.path.join(plot_path, 'pattern_stats.csv')

do_plot = False
pat_columns = ['name', 'step', 'active_step', 'pat_mean', 'pat_std', 'h_last']

# utilities

mpl.rcParams['font.size'] = 18


# build scenario

if __name__ == '__main__':

  pat_stats_set = []

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
    if h_index.shape[0] > 1:
      h_index[0] = h_index[1]
    s_index = S_stats['s-index']
    p_index = 1 - np.array(S_stats['distance-worst-o'])
    g_index = 1 - np.array(S_stats['distance-worst-s'])
    
    # calculate stats
    
    p_index_resmpl = interp1d(S_stat_steps, p_index, kind='linear')(S_data_steps)
    g_index_resmpl = interp1d(S_stat_steps, g_index, kind='linear')(S_data_steps)
    
    g_mask = g_index_resmpl <= np.max(g_index) * 0.95
    pat_diff = (h_index - p_index_resmpl)[g_mask]
    p_last = p_index[-1]
    pat_mean = np.mean(pat_diff)
    pat_std = np.std(pat_diff)
    
    pat_stats = [
      scenario_name, 
      S.steps, np.sum(g_mask, dtype=int), 
      pat_mean, pat_std, p_last
    ]
    pat_stats_set.append(pat_stats)
    
    pat_stats_df = pd.DataFrame(pat_stats_set, columns=pat_columns)
    pat_stats_df.to_csv(pat_csv_path, index=False, encoding='utf-8-sig')

    if not do_plot:
      continue
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
