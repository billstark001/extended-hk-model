from typing import List, Tuple, cast, Any

import os

import json
import pickle
import importlib
import dataclasses

import numpy as np
import networkx as nx
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scipy.interpolate import interp1d


from utils.stat import area_under_curve, first_more_or_equal_than
import works.detail.simulate as p
from works.gradation.stat import ScenarioPatternRecorder
import utils.plot as _p
importlib.reload(_p)


# parameters

scenario_base_path = './run3'
plot_path = './fig3'

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

pat_csv_path = os.path.join(plot_path, 'pattern_stats.json')


active_threshold = 0.98
min_inactive_value = 0.75

# utilities

mpl.rcParams['font.size'] = 18


# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"

if __name__ == '__main__':

  for scenario_name, r, d, g in p.params_arr: # tqdm(p.params_arr, bar_format=short_progress_bar):

    # load scenario
    
    scenario_path = os.path.join(scenario_base_path, scenario_name + '_record.pkl')
    if not os.path.exists(scenario_path):
      continue
    
    snapshot_path = os.path.join(plot_path, scenario_name + '_snapshot.png')
    if os.path.exists(snapshot_path):
      continue

    with open(scenario_path, 'rb') as f:
      S_metadata, S_stats, S_agent_stats = pickle.load(f)
      
    S_stat_steps = np.array(S_stats['step'], dtype=int)
      
    nr_agents = S_agent_stats['nr_agents']
      
    n_n = nr_agents[..., 0]
    n_rec = nr_agents[..., 1]
    n_agents = n_n.shape[1]
    
    S_data_steps = S_agent_stats['step']
    
    # collect indices

    n_edges = S_metadata['n_edges']
    h_index = np.mean(n_n / n_edges[np.newaxis, :], axis=1)
    if h_index.shape[0] > 1:
      h_index[0] = h_index[1]
    s_index = S_stats['s-index']
    p_index = 1 - np.array(S_stats['distance-worst-o'])
    g_index = 1 - np.array(S_stats['distance-worst-s'])

    # gradation index

    p_index_resmpl = interp1d(S_stat_steps, p_index,
                              kind='linear')(S_data_steps)
    g_index_resmpl = interp1d(S_stat_steps, g_index,
                              kind='linear')(S_data_steps)

    active_step=int(first_more_or_equal_than(
      g_index_resmpl, 
      np.max([np.max(g_index) * active_threshold, min_inactive_value])
    ))
    active_step_threshold = g_index_resmpl[active_step - 1]
    
    pat_area_hp=area_under_curve([p_index_resmpl, h_index])
    pat_area_ps=area_under_curve([s_index, p_index])
    
    # get sample
    
    step_indices = np.array([
      np.argmin(np.abs(S_stat_steps - active_step * k)) \
        for k in np.arange(0, 1 + 1/16, 1/16)
    ]) 
    steps_smpl = S_stat_steps[step_indices]
    opinions_smpl = [S_stats['layout-opinion'][x] for x in step_indices]
    graphs_smpl = [S_stats['layout-graph'][x] for x in step_indices]
    
    # plot
    
    fig, all_axes = cast(Tuple[Figure, List[Axes]],
                       _p.plt_figure(n_col=5, total_width=20))
    all_indices = [0, 3, 7, 11, 15]

    plotted_indices = set()
    pos = None
    for i_ax, i in tqdm(enumerate(all_indices), bar_format=short_progress_bar):
      if i in plotted_indices:
        continue
      step = S_stat_steps[i]
      opinion = opinions_smpl[i]
      graph = graphs_smpl[i]
      pos = nx.spring_layout(graph, pos=pos)
      _p.plot_network_snapshot(pos, opinion, graph, all_axes[i_ax], step)
      plotted_indices.add(i)
      
    title = f'decay: {d:.4f}, rewiring: {r:.4f}, gradation: {pat_area_hp:.6f}'
    plt.title(title)

    plt.savefig(snapshot_path, bbox_inches='tight')
    plt.close()

    print(scenario_name)
