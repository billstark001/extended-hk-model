from numpy.typing import NDArray

import numpy as np
from scipy.stats import pearsonr
import networkx as nx

from scipy.interpolate import interp1d

from utils.context import Context
from utils.stat import area_under_curve, compress_array_to_b64, first_more_or_equal_than


c = Context(
    ignore_prefix='get_'
)
c.set_state(
    active_threshold=0.98,
    min_inactive_value=0.75,
)


_b = compress_array_to_b64
dummy_comp_arr = _b(np.zeros((0,), dtype=int))


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
    A_triads[A == 0] = 0
    n_triads = np.sum(A_triads)

  return A_fw, n_triads, A_triads


@c.selector
def get_agent_stats_step(agent_stats):
  return np.array(agent_stats['step'], dtype=int)


@c.selector(('n_neighbor', 'n_recommended', 'n_agents'))
def parse_numbers(agent_stats):
  nr_agents = agent_stats['nr_agents']

  n_n = nr_agents[..., 0]
  n_rec = nr_agents[..., 1]
  n_agents = n_n.shape[1]

  return n_n, n_rec, n_agents


@c.selector
def get_opinion(agent_stats):
  return agent_stats['cur_opinion']


@c.selector((
    'event_step', 'event_step_based_on', 'event_agent',
    'event_unfo', 'event_fo',
    'event_op_cur', 'event_op_unfo', 'event_op_fo'
))
def parse_events(
    agent_stats, agent_stats_step,
    n_agents
):

  S_data_steps_mat = np.repeat(
      agent_stats_step.reshape((-1, 1)), n_agents, axis=1)
  S_data_agents_mat = np.repeat(np.arange(n_agents, dtype=int).reshape(
      (1, -1)), agent_stats_step.size, axis=0)
  opinion = agent_stats['cur_opinion']

  follow_events = agent_stats['follow_event']
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

  return event_step, event_step_based_on, event_agent, \
      event_unfo, event_fo, \
      event_op_cur, event_op_unfo, event_op_fo


@c.selector
def get_model_stats_step(model_stats):
  return np.array(model_stats['step'], dtype=int)


@c.selector
def get_total_steps(model_metadata):
  return model_metadata['total_steps']


@c.selector(('h_index', 's_index', 'p_index', 'g_index'))
def get_indices(n_neighbor, model_stats, model_metadata):
  n_edges = model_metadata['n_edges']
  h_index = np.mean(n_neighbor / n_edges[np.newaxis, :], axis=1)
  if h_index.shape[0] > 1:
    h_index[0] = h_index[1]
  s_index = model_stats['s-index']
  p_index = 1 - np.array(model_stats['distance-worst-o'])
  g_index = 1 - np.array(model_stats['distance-worst-s'])

  return h_index, s_index, p_index, g_index


@c.selector(('s_index_resmpl', 'p_index_resmpl', 'g_index_resmpl'))
def get_indices_resmpl(model_stats_step, agent_stats_step, s_index, p_index, g_index):
  s_index_resmpl = interp1d(model_stats_step, s_index,
                            kind='linear')(agent_stats_step)
  p_index_resmpl = interp1d(model_stats_step, p_index,
                            kind='linear')(agent_stats_step)
  g_index_resmpl = interp1d(model_stats_step, g_index,
                            kind='linear')(agent_stats_step)
  return s_index_resmpl, p_index_resmpl, g_index_resmpl


@c.selector
def get_gradation_index_hp(p_index_resmpl, h_index):
  return area_under_curve([p_index_resmpl, h_index])


@c.selector(('active_step', 'active_step_threshold', 'g_index_active', 'g_index_mean_active'))
def calc_active_step(g_index, g_index_resmpl, active_threshold, min_inactive_value):
  active_step = int(first_more_or_equal_than(
      g_index_resmpl,
      np.max([np.max(g_index) * active_threshold, min_inactive_value])
  ))
  active_step_threshold = g_index_resmpl[active_step - 1]
  g_index_active = g_index_resmpl[:active_step]
  g_index_mean_active = np.mean(g_index_active)

  return active_step, active_step_threshold, g_index_active, g_index_mean_active


@c.selector
def get_n_triads(model_stats):
  _, n_triads, __ = get_adj_fw_triads(
      model_stats['layout-graph'][-1], fw=False)
  n_triads = int(n_triads)
  return n_triads


@c.selector
def get_gradation_index_pat_diff(h_index, p_index_resmpl, active_step):
  return (h_index - p_index_resmpl)[:active_step]


@c.selector(('opinion_last', 'opinion_last_mean', 'opinion_last_diff'))
def calc_opinion_last(opinion):
  opinion_last = opinion[-1]
  opinion_last_mean = np.mean(opinion_last)
  opinion_last_diff = \
      np.mean(opinion_last[opinion_last > opinion_last_mean]) - \
      np.mean(opinion_last[opinion_last <= opinion_last_mean])
  return opinion_last, opinion_last_mean, opinion_last_diff


@c.selector
def get_in_degree(model_stats):
  in_degree = [model_stats[x][-1]
               for x in ['in-degree-alpha', 'in-degree-p-value', 'in-degree-R']]
  in_degree = [None if not np.isfinite(x) else x for x in in_degree]
  return in_degree


@c.selector(('smpl_pearson_rel', 'smpl_rec_dis_network', 'smpl_rec_concordant_n'))
def get_micro_level_stats(
    model_stats, model_stats_step, active_step,
    n_recommended, agent_stats_step,
    event_step, event_agent, event_fo, event_op_fo,
):
  step_indices = np.array([
      np.argmin(np.abs(model_stats_step - active_step * k))
      for k in np.arange(0, 1 + 1/16, 1/16)
  ])
  smpl_steps = model_stats_step[step_indices]
  opinions_smpl = [model_stats['layout-opinion'][x] for x in step_indices]
  graphs_smpl = [model_stats['layout-graph'][x] for x in step_indices]
  opinion_diff_smpl = [
      np.abs(x.reshape((1, -1)) - x.reshape((-1, 1))) for x in opinions_smpl]
  graph_dis_smpl_ = [get_adj_fw_triads(x, triads=False) for x in graphs_smpl]
  graph_dis_smpl = [x[0] for x in graph_dis_smpl_]

  mask_mat = np.ones((400, 400), dtype=bool)
  mask_mat = np.triu(mask_mat, 1)
  masks_smpl = [np.logical_and(np.isfinite(x), mask_mat)
                for x in graph_dis_smpl]

  point_set_smpl = [
      np.concatenate([
          opinion_diff_smpl[i][masks_smpl[i]].reshape((1, -1)),
          graph_dis_smpl[i][masks_smpl[i]].reshape((1, -1))], axis=0)
      for i in range(len(masks_smpl))
  ]

  # mutual_info_smpl = [
  #   mutual_info_score(x[0], x[1]) for x in point_set_smpl
  # ]
  smpl_pearson_rel = [
      pearsonr(x[0], x[1]) for x in point_set_smpl
  ]
  smpl_pearson_rel = [None if not np.isfinite(
      x.statistic) else x.statistic for x in smpl_pearson_rel]

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

  n_rec_m = np.mean(n_recommended, axis=1)
  for i, smpl_step in enumerate(smpl_steps):
    step_mask = agent_stats_step > smpl_step
    if i != smpl_steps.size - 1:
      step_mask = np.logical_and(
          step_mask, agent_stats_step < smpl_steps[i + 1])
    n_rec_mm = np.mean(n_rec_m[step_mask])
    smpl_rec_concordant_n.append(n_rec_mm if np.isfinite(n_rec_mm) else None)

  return smpl_pearson_rel, smpl_rec_dis_network, smpl_rec_concordant_n


G, state_vars, cycles = c.get_dep_graph()

if cycles:
  raise ValueError('Cyclic dependencies detected: ' +
                   ','.join([str(c) for c in cycles]))
