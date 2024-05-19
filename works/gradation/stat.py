from typing import cast, List, Tuple, Any
from numpy.typing import NDArray

import os

import json
import pickle
import importlib
import dataclasses

import numpy as np
from scipy.stats import pearsonr
import networkx as nx
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes

from scipy.interpolate import interp1d


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

  cluster: float = 0
  triads: int = 0
  triads2: int = 0
  opinion_diff: float = 0
  in_degree: Tuple[float, float, float] = dataclasses.field(default_factory=lambda: [-1, -1, -1])
  
  event_step: str = dummy_comp_arr
  event_unfollow: str = dummy_comp_arr
  event_follow: str = dummy_comp_arr
  
  smpl_pearson_rel: List[float] = dataclasses.field(default_factory=list)
  smpl_mutual_info: List[float] = dataclasses.field(default_factory=list)
  smpl_rec_dis_network: List[float] = dataclasses.field(default_factory=list)
  smpl_rec_concordant_n: List[float] = dataclasses.field(default_factory=list)


active_threshold = 0.98
min_inactive_value = 0.75

# utilities

mpl.rcParams['font.size'] = 18


# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"

# from nx.floyd_warshall_numpy
def floyd_warshall_weight_matrix(A: NDArray):
  n, m = A.shape
  np.fill_diagonal(A, 0)  # diagonal elements should be zero
  for i in range(n):
    # The second term has the same shape as A due to broadcasting
    A = np.minimum(A, A[i, :][np.newaxis, :] + A[:, i][:, np.newaxis])
  return A

def get_adj_fw_triads(G: nx.DiGraph, fw=True, triads=True):
  A = nx.to_numpy_array(G, multigraph_weight=min, nonedge=np.inf)
  A_graph = np.minimum(A, A.T)
  
  # floyd_warshall
  A_fw = floyd_warshall_weight_matrix(A_graph) if fw else A
  
  n_triads = 0
  A_triads = A
  if triads:
    A[np.isinf(A)] = 0
    A2 = A @ A
    A_triads = np.copy(A2)
    A2[A == 0] = 0
    n_triads = np.sum(A_triads)
  
  return A_fw, n_triads, A_triads

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
    
    scenario_path = os.path.join(scenario_base_path, scenario_name + '_record.pkl')
    if not os.path.exists(scenario_path):
      continue

    with open(scenario_path, 'rb') as f:
      S_metadata, S_stats, S_agent_stats = pickle.load(f)
      
      
    nr_agents = S_agent_stats['nr_agents']
      
    n_n = nr_agents[..., 0]
    n_rec = nr_agents[..., 1]
    n_agents = n_n.shape[1]
    
    S_data_steps = S_agent_stats['step']
    
    S_data_steps_mat = np.repeat(S_data_steps.reshape((-1, 1)), n_agents, axis=1)
    S_data_agents_mat = np.repeat(np.arange(n_agents, dtype=int).reshape((1, -1)), S_data_steps.size, axis=0)
    opinion = S_agent_stats['cur_opinion']
    
    follow_events = S_agent_stats['follow_event']
    has_follow_event = follow_events[..., 0] == 1
    unfollowed = follow_events[..., 1]
    followed = follow_events[..., 2]
    
    event_step = S_data_steps_mat[has_follow_event]
    event_step_based_on = S_data_steps_mat[has_follow_event] - 1
    assert event_step_based_on.min() >= 0
    event_agent = S_data_agents_mat[has_follow_event]
    event_unfo = unfollowed[has_follow_event]
    event_fo = followed[has_follow_event]
    
    event_op_cur = opinion[event_step, event_agent]
    event_op_unfo = np.abs(opinion[event_step, event_unfo] - event_op_cur)
    event_op_fo = np.abs(opinion[event_step, event_fo] - event_op_cur)

    S_stat_steps = np.array(S_stats['step'], dtype=int)

    # collect indices

    n_edges = S_metadata['n_edges']
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
        
    _, n_triads, __ = get_adj_fw_triads(S_stats['layout-graph'][-1], fw=False)
    n_triads = int(n_triads)
        
    # calculate micro-level stats
    
    step_indices = np.array([
      np.argmin(np.abs(S_stat_steps - active_step * k)) \
        for k in np.arange(0, 1 + 1/16, 1/16)
    ]) 
    smpl_steps = S_stat_steps[step_indices]
    opinions_smpl = [S_stats['layout-opinion'][x] for x in step_indices]
    graphs_smpl = [S_stats['layout-graph'][x] for x in step_indices]
    opinion_diff_smpl = [np.abs(x.reshape((1, -1)) - x.reshape((-1, 1))) for x in opinions_smpl]
    graph_dis_smpl_ = [get_adj_fw_triads(x, triads=False) for x in graphs_smpl]
    graph_dis_smpl = [x[0] for x in graph_dis_smpl_]
    
    mask_mat = np.ones((400, 400), dtype=bool)
    mask_mat = np.triu(mask_mat, 1)
    masks_smpl = [np.logical_and(np.isfinite(x), mask_mat) for x in graph_dis_smpl]
    
    point_set_smpl = [
      np.concatenate([
        opinion_diff_smpl[i][masks_smpl[i]].reshape((1, -1)), 
        graph_dis_smpl[i][masks_smpl[i]].reshape((1, -1))], axis=0) \
        for i in range(len(masks_smpl))
    ]
    
    # mutual_info_smpl = [
    #   mutual_info_score(x[0], x[1]) for x in point_set_smpl
    # ]
    pearson_rel_smpl = [
      pearsonr(x[0], x[1]) for x in point_set_smpl
    ]
    pearson_rel_smpl = [None if not np.isfinite(x.statistic) else x.statistic for x in pearson_rel_smpl]
    
    # explanation of recommended distance
    
    smpl_rec_dis_network = []
    
    
    for i, smpl_step in enumerate(smpl_steps):
      event_mask = event_step > smpl_step
      if i != smpl_steps.size - 1:
        event_mask = np.logical_and(event_mask, event_step <= smpl_steps[i + 1])
      graph_dis = graph_dis_smpl[i]
      
      e_agents = event_agent[event_mask]
      e_targets = event_fo[event_mask]
      dis = graph_dis[e_agents, e_targets]
      dis_op = event_op_fo[event_mask]
      dis_mask = np.logical_not(np.isinf(dis))
      dis = dis[dis_mask]
      dis_op = dis_op[dis_mask]
      smpl_rec_dis_network.append(
        [e_agents.size, dis.size, np.mean(dis) if dis.size > 0 else None]
      )
      
    # explanation of recommended opinion distance
    
    smpl_rec_concordant_n = []
    
    n_rec_m = np.mean(n_rec, axis=1)
    for i, smpl_step in enumerate(smpl_steps):
      step_mask = S_data_steps > smpl_step
      if i != smpl_steps.size - 1:
        step_mask = np.logical_and(step_mask, S_data_steps < smpl_steps[i + 1])
      n_rec_mm = np.mean(n_rec_m[step_mask])
      smpl_rec_concordant_n.append(n_rec_mm if np.isfinite(n_rec_mm) else None)
        
    # in_degree
    in_degree = [S_stats[x][-1]
                   for x in ['in-degree-alpha', 'in-degree-p-value', 'in-degree-R']]
    in_degree = [None if not np.isfinite(x) else x for x in in_degree]
        
    # create json

    total_steps = S_metadata['total_steps']
    pat_stats = ScenarioPatternRecorder(
        name=scenario_name,
        step=total_steps,
        
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
        
        # cluster=S_stats['cluster'][-1],
        # triads=S_stats['triads'][-1],
        triads2 = n_triads,
        
        in_degree=in_degree,
        opinion_diff=opinion_last_diff if np.isfinite(
            opinion_last_diff) else -1,
        
        # event_step=_b(event_step),
        # event_unfollow=_b(event_op_unfo),
        # event_follow=_b(event_op_fo),
        
        # smpl_mutual_info=mutual_info_smpl,
        smpl_pearson_rel=pearson_rel_smpl,
        smpl_rec_dis_network=smpl_rec_dis_network,
        smpl_rec_concordant_n=smpl_rec_concordant_n,
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
    ax.set_xlabel(f'step (total: {total_steps})')

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
