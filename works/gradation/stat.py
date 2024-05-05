from typing import cast, List, Tuple, Any, Optional
from numpy.typing import NDArray

import os

import base64
import json
import pickle
import importlib
import dataclasses

import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
import networkx as nx
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes

from scipy.interpolate import interp1d

from base import Scenario

from utils.plot import plot_network_snapshot, plt_figure
from utils.stat import area_under_curve, compress_array_to_b64, first_more_or_equal_than
import works.gradation.simulate as p
import utils.plot as _p
importlib.reload(_p)


# parameters

scenario_base_path = './run2'
plot_path = './fig2'

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

pat_csv_path = os.path.join(plot_path, 'pattern_stats.json')

do_plot = False
do_plot_layout = False

_b = compress_array_to_b64
dummy_comp_arr = _b(np.zeros((0,), dtype=int))

@dataclasses.dataclass
class ScenarioPatternRecorder:
  name: str
  step: int
  
  active_step: int
  active_step_threshold: float
  g_index_mean_active: float

  p_last: float
  s_last: float
  h_last: float

  pat_abs_mean: float
  pat_abs_std: float

  pat_area_hp: float
  pat_area_ps: float

  cluster: float
  triads: float
  opinion_diff: float
  in_degree: Tuple[float, float, float] = dataclasses.field(default_factory=lambda: [-1, -1, -1])
  
  event_step: str = dummy_comp_arr
  event_unfollow: str = dummy_comp_arr
  event_follow: str = dummy_comp_arr
  
  smpl_pearson_rel: List[float] = dataclasses.field(default_factory=list)
  smpl_mutual_info: List[float] = dataclasses.field(default_factory=list)


active_threshold = 0.98
min_inactive_value = 0.75

# utilities

mpl.rcParams['font.size'] = 18


# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


if __name__ == '__main__':

  pat_stats_set: List[ScenarioPatternRecorder] = []
  processed_data = {}

  if os.path.exists(pat_csv_path):
    with open(pat_csv_path, 'r', encoding='utf8') as fp:
      lst = json.load(fp)
      for d in lst:
        processed_data[d['name']] = d

  unsaved = False

  def save_stats():
    global unsaved
    with open(pat_csv_path, 'w', encoding='utf8') as f:
      json.dump(pat_stats_set, f, indent=2, ensure_ascii=False)
    unsaved = False

  for scenario_name, r, d, g in tqdm(p.params_arr, bar_format=short_progress_bar):

    if scenario_name in processed_data:
      pat_stats_set.append(processed_data[scenario_name])
      continue

    if unsaved:
      save_stats()

    # load scenario

    params = p.gen_params(r, d, g)
    p.sim_p_standard.model_stat_collectors = p.stat_collectors_f(layout=True)
    S = Scenario(p.network_provider, params, p.sim_p_standard)

    scenario_path = os.path.join(scenario_base_path, scenario_name + '.pkl')
    if not os.path.exists(scenario_path):
      continue

    with open(scenario_path, 'rb') as f:
      dump_data = pickle.load(f)
    S.load(*dump_data)

    # collect stats

    if S.steps not in S.stats:
      S.add_model_stats()
      
      
    S_agent_stats = S.generate_agent_stats()
    n_n = S_agent_stats['n_neighbor']
    n_agents = n_n.shape[1]
    
    S_data_steps = S_agent_stats['step']
    
    S_data_steps_mat = np.repeat(S_data_steps.reshape((-1, 1)), n_agents, axis=1)
    S_data_agents_mat = np.repeat(np.arange(n_agents, dtype=int).reshape((1, -1)), S_data_steps.size, axis=0)
    opinion = S_agent_stats['cur_opinion']
    
    has_follow_event = S_agent_stats['has_follow_event']
    unfollowed = S_agent_stats['unfollowed']
    followed = S_agent_stats['followed']
    
    event_step = S_data_steps_mat[has_follow_event]
    event_agent = S_data_agents_mat[has_follow_event]
    event_unfo = unfollowed[has_follow_event]
    event_fo = followed[has_follow_event]
    
    event_op_cur = opinion[event_step, event_agent]
    event_op_unfo = np.abs(opinion[event_step, event_unfo] - event_op_cur)
    event_op_fo = np.abs(opinion[event_step, event_fo] - event_op_cur)

    S_stats = S.generate_model_stats()
    S_stat_steps = np.array(S_stats['step'], dtype=int)

    # collect indices

    n_edges = np.array(
        sorted(list(S.model.graph.out_degree), key=lambda x: x[0]))[:, 1]
    h_index = np.mean(n_n / n_edges[np.newaxis, :], axis=1)
    if h_index.shape[0] > 1:
      h_index[0] = h_index[1]
    s_index = S_stats['s-index']
    p_index = 1 - np.array(S_stats['distance-worst-o'])
    g_index = 1 - np.array(S_stats['distance-worst-s'])

    # calculate stats

    p_index_resmpl = interp1d(S_stat_steps, p_index,
                              kind='linear')(S_data_steps)
    g_index_resmpl = interp1d(S_stat_steps, g_index,
                              kind='linear')(S_data_steps)

    active_step=int(first_more_or_equal_than(
      g_index_resmpl, 
      np.max([np.max(g_index) * active_threshold, min_inactive_value])
    ))
    active_step_threshold = g_index_resmpl[active_step - 1]
    g_index_active = g_index_resmpl[:active_step]
    g_index_mean_active = np.mean(g_index_active)
    
    pat_diff = (h_index - p_index_resmpl)[:active_step]

    opinion_last = opinion[-1]
    opinion_last_mean = np.mean(opinion_last)
    opinion_last_diff = \
        np.mean(opinion_last[opinion_last > opinion_last_mean]) - \
        np.mean(opinion_last[opinion_last <= opinion_last_mean])
        
    # calculate micro-level stats
    
    step_indices = np.array([
      np.argmin(np.abs(S_stat_steps - active_step * k)) \
        for k in [1/16, 1/8, 1/4, 1/2, 3/4]
    ])
    smpl_steps = S_stat_steps[step_indices]
    layouts_smpl = [S_stats['layout'][x] for x in step_indices]
    opinions_smpl = [x[1] for x in layouts_smpl]
    graphs_smpl = [x[2] for x in layouts_smpl]
    opinion_diff_smpl = [np.abs(x.reshape((1, -1)) - x.reshape((-1, 1))) for x in opinions_smpl]
    graph_dis_smpl = [nx.floyd_warshall_numpy(x.to_undirected()) for x in graphs_smpl]
    
    mask_mat = np.ones((400, 400), dtype=bool)
    mask_mat = np.triu(mask_mat, 1)
    masks_smpl = [np.logical_and(np.isfinite(x), mask_mat) for x in graph_dis_smpl]
    
    point_set_smpl = [
      np.concatenate([
        opinion_diff_smpl[i][masks_smpl[i]].reshape((1, -1)), 
        graph_dis_smpl[i][masks_smpl[i]].reshape((1, -1))], axis=0) \
        for i in range(len(masks_smpl))
    ]
    
    mutual_info_smpl = [
      mutual_info_score(x[0], x[1]) for x in point_set_smpl
    ]
    pearson_rel_smpl = [
      pearsonr(x[0], x[1]).statistic for x in point_set_smpl
    ]
    
    # explanation of recommended distance
    
    # TODO
    
        
    # create json

    pat_stats = ScenarioPatternRecorder(
        name=scenario_name,
        step=S.steps,
        
        active_step=active_step,
        active_step_threshold=active_step_threshold,
        g_index_mean_active=g_index_mean_active,
        
        p_last=p_index[-1],
        h_last=h_index[-1],
        s_last=s_index[-1],
        
        pat_abs_mean=np.mean(pat_diff),
        pat_abs_std=np.std(pat_diff),
        pat_area_hp=area_under_curve([p_index_resmpl, h_index]),
        pat_area_ps=area_under_curve([s_index, p_index]),
        
        cluster=S_stats['cluster'][-1],
        triads=S_stats['triads'][-1],
        
        in_degree=[S_stats[x][-1]
                   for x in ['in-degree-alpha', 'in-degree-p-value', 'in-degree-R']],
        opinion_diff=opinion_last_diff if np.isfinite(
            opinion_last_diff) else -1,
        
        event_step=_b(event_step),
        event_unfollow=_b(event_op_unfo),
        event_follow=_b(event_op_fo),
        
        smpl_mutual_info=mutual_info_smpl,
        smpl_pearson_rel=pearson_rel_smpl,
    )

    pat_stats_set.append(dataclasses.asdict(pat_stats))
    save_stats()

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

    plt.savefig(os.path.join(plot_path, scenario_name +
                '_stats.png'), bbox_inches='tight')
    plt.close()

    # plot networks

    if not do_plot_layout:
      continue

    layouts = S_stats['layout']

    _, (r1, r2) = cast(Tuple[Any, List[List[Axes]]],
                       plt_figure(n_col=4, n_row=2))
    all_axes = list(r1) + list(r2)
    all_indices = np.array(np.linspace(
        0, len(layouts) - 1, len(all_axes)), dtype=int)

    plotted_indices = set()
    for i_ax, i in enumerate(all_indices):
      if i in plotted_indices:
        continue
      step = S_stat_steps[i]
      pos, opinion, graph = layouts[i]
      plot_network_snapshot(pos, opinion, graph, all_axes[i_ax], step)
      plotted_indices.add(i)

    plt.savefig(os.path.join(plot_path, scenario_name +
                '_snapshot.png'), bbox_inches='tight')
    plt.close()

    print(scenario_name)

  if unsaved:
    save_stats()
