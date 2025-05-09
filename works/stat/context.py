from numpy.typing import NDArray

import numpy as np
from scipy.stats import skew, kurtosis
import networkx as nx


from result_interp.parse_events_db import get_events_by_step_range
from result_interp.record import RawSimulationRecord
from stats.distance_c import DistanceCollectorContinuous
from utils.context import Context
from utils.stat import adaptive_discrete_sampling, area_under_curve, first_more_or_equal_than, merge_data_with_axes


c = Context(
    ignore_prefix='get_'
)
c.set_state(
    active_threshold=0.98,
    min_inactive_value=0.75,
)

# from nx.floyd_warshall_numpy


def floyd_warshall_weight_matrix(A: NDArray):
  n, m = A.shape
  np.fill_diagonal(A, 0)  # diagonal elements should be zero
  for i in range(n):
    # The second term has the same shape as A due to broadcasting
    A = np.minimum(A, A[i, :][np.newaxis, :] + A[:, i][:, np.newaxis])
  return A


def get_fw_stats(A_in: NDArray):
  # A_in: adjacency boolean array (nonedge = 0)
  A = np.copy(A_in).astype(float)
  A[A > 1] = 1
  A[A == 0] = np.inf
  A_graph = np.minimum(A, A.T)
  A_fw = floyd_warshall_weight_matrix(A_graph)
  return A_fw


def get_triads_stats(A: NDArray):
  # A: adjacency boolean array (nonedge = 0)
  A2 = A @ A
  A_triads = np.copy(A2)
  A_triads[A == 0] = 0
  n_triads = np.sum(A_triads)

  return n_triads, A_triads


def get_opinion_diff_vars(adj_mat: NDArray, opinion: NDArray):
  out_deg = np.sum(adj_mat, axis=1).reshape(-1, 1)
  o = opinion.reshape(-1, 1)
  o_adj_sum = adj_mat @ o
  o_cur_sum = out_deg * o
  o2_adj_sum = adj_mat @ (o ** 2)
  o2_cur_sum = out_deg * (o ** 2)

  var_term_1_sum = o2_adj_sum + o2_cur_sum - 2 * o_adj_sum * o
  var_term_2_sum = o_adj_sum - o_cur_sum

  var = var_term_1_sum / out_deg - (var_term_2_sum / out_deg) ** 2

  return var.flatten()

# bimodality coefficient


def bimodality_coefficient(data: NDArray):
  """
  Calculate the Bimodality Coefficient (BC) for a given dataset.

  Parameters:
  data (numpy.array): Input data

  Returns:
  float: Bimodality Coefficient
  """
  n = data.size
  m3 = skew(data, bias=False)
  m4 = kurtosis(data, bias=False)
  numerator = m3 ** 2 + 1
  denominator = m4 + 3 * ((n-1)**2) / ((n-2)*(n-3))

  bc = float(numerator / denominator)
  return bc


def get_bc_hom(G: nx.DiGraph, opinion: NDArray):
  edges = np.array(G.edges())
  paired_opinion = opinion[edges]
  rotated_opinion = np.sum(paired_opinion, axis=1) / 2
  return bimodality_coefficient(rotated_opinion)


@c.selector
def get_rewiring_events(
    scenario_record: RawSimulationRecord
):
  rewiring_events = get_events_by_step_range(
      scenario_record.events_db,
      0, scenario_record.max_step + 1,
      "Rewiring"
  )
  event_step = np.array([x.step for x in rewiring_events])
  # TODO add other statistics when necessary
  return event_step


@c.selector
def get_total_steps(scenario_record: RawSimulationRecord):
  return scenario_record.max_step

# region indices


distance_collector = DistanceCollectorContinuous(
    use_js_divergence=True,
    min_bandwidth=0.01,
    node_cnt=500,
)


def err_func_distance(x1, x2, x3, t3):
  return max(
      abs(x1[i] * (1 - t3) + x2[i] * t3 - x3[i])
      for i in range(4)
  )


def calc_distance(rec: RawSimulationRecord, step: int):
  #   print('dis', step)
  graph = rec.get_graph(step)
  opinion = rec.opinions[step]

  dis_res = distance_collector.collect('d', graph, opinion)

  d_rand_o = dis_res['d-rand-o']
  d_rand_s = dis_res['d-rand-s']
  d_worst_o = 1 - dis_res['d-worst-o']
  d_worst_s = 1 - dis_res['d-worst-s']

  return d_rand_o, d_rand_s, d_worst_o, d_worst_s


def calc_homophily(rec: RawSimulationRecord, step: int):
  f = rec.followers
  f_slice = rec.agent_numbers[step, :, 0]
  return np.mean(f_slice / f, dtype=float)


@c.selector
def get_indices(scenario_record: RawSimulationRecord):
  g_distance = adaptive_discrete_sampling(
      lambda x: calc_distance(scenario_record, x),
      0.01,
      0, scenario_record.max_step, max_interval=512,
      err_func=err_func_distance,
  )
  g_homophily = adaptive_discrete_sampling(
      lambda x: calc_homophily(scenario_record, x), 
      0.01,
      0, scenario_record.max_step, max_interval=512,
  )
  x_indices_raw, (y_dist, y_homo) = merge_data_with_axes(
      g_distance, g_homophily,
  )

  x_indices = x_indices_raw.tolist()
  h_index = y_homo.tolist()
  p_index = y_dist[:, 2].tolist()  # d-worst-o
  g_index = y_dist[:, 3].tolist()  # d-worst-s

  return x_indices, h_index, p_index, g_index


# endregion

@c.selector
def get_gradation_index_hp(p_index, h_index):
  return area_under_curve(np.array([p_index, h_index]))


@c.selector
def calc_active_step(x_indices, g_index, active_threshold, min_inactive_value):
  active_step_index = int(first_more_or_equal_than(
      np.array(g_index),
      np.max([np.max(g_index) * active_threshold, min_inactive_value])
  ))
  active_step = x_indices[active_step_index]
  active_step_threshold = g_index[active_step_index - 1]
  g_index_active = g_index[:active_step_index]
  g_index_mean_active = np.mean(g_index_active)

  return active_step, active_step_threshold, \
      g_index_active, g_index_mean_active


@c.selector
def get_gradation_index_pat_diff(h_index, p_index, active_step):
  return (h_index - p_index)[:active_step]


@c.selector
def calc_opinion_last(scenario_record: RawSimulationRecord):
  opinion_last = scenario_record.opinions[scenario_record.max_step]
  opinion_last_mean = float(np.mean(opinion_last))
  opinion_last_diff = float(\
      np.mean(opinion_last[opinion_last > opinion_last_mean]) - \
      np.mean(opinion_last[opinion_last <= opinion_last_mean]))
  return opinion_last, opinion_last_mean, opinion_last_diff


@c.selector
def get_opinion_diff(scenario_record: RawSimulationRecord):
  opinion = scenario_record.opinions
  opinion_diff = np.copy(opinion)
  opinion_diff[1:] -= opinion[:-1]
  if opinion_diff.shape[0] > 1:
    opinion_diff[0] = opinion_diff[1]
  return opinion_diff


@c.selector
def get_opinion_diff_mean(opinion_diff: NDArray, scenario_record: RawSimulationRecord):
  odm_raw = np.abs(opinion_diff).mean(axis=1)
  x_opinion_diff_mean, opinion_diff_mean_smpl = adaptive_discrete_sampling(
    lambda x: float(odm_raw[x]),
    0.01,
    0, scenario_record.max_step, max_interval=512,
  )
  return x_opinion_diff_mean, opinion_diff_mean_smpl


# @c.selector
# def get_in_degree(model__stats):
#   in_degree = [model__stats[x][-1]
#                for x in ['in-degree-alpha', 'in-degree-p-value', 'in-degree-R']]
#   in_degree = [None if not np.isfinite(x) else x for x in in_degree]
#   return in_degree


@c.selector
def get_n_triads_and_bc_hom(scenario_record: RawSimulationRecord):
  last_graph = scenario_record.get_graph(scenario_record.max_step)
  last_opinion = scenario_record.opinions[scenario_record.max_step]
  adj_mat = nx.to_numpy_array(last_graph, multigraph_weight=min)
  n_triads, _ = get_triads_stats(adj_mat)
  n_triads = int(n_triads)

  bc_hom_last = get_bc_hom(last_graph, last_opinion)

  return n_triads, bc_hom_last

# adjacency variances


@c.selector
def get_mean_vars_x_and_smpl(
    scenario_record: RawSimulationRecord,
):
  def get_vars(step: int):
    g = scenario_record.get_graph(step)
    o = scenario_record.opinions[step]
    A = nx.to_numpy_array(g, multigraph_weight=min)  # adjacency matrix
    vars = get_opinion_diff_vars(A, o)
    return np.mean(vars)

  x_mean_vars, mean_vars_smpl = adaptive_discrete_sampling(
      get_vars,
      0.01,
      0, scenario_record.max_step, max_interval=512,
  )

  return x_mean_vars, mean_vars_smpl


@c.selector
def get_bc_hom_x_and_smpl(scenario_record: RawSimulationRecord):
  def get_bc_hom_step(step: int):
    g = scenario_record.get_graph(step)
    o = scenario_record.opinions[step]
    ret = get_bc_hom(g, o)
    if np.isnan(ret):
      return None
    return ret

  def err_bc_hom(x1, x2, x3, t3):
    if x1 is None and x2 is None and x3 is None:
      return 0
    if x1 is None or x2 is None or x3 is None:
      return 1
    ret = x1 * (1 - t3) + x2 * t3 - x3
    return ret

  x_bc_hom, bc_hom_smpl = adaptive_discrete_sampling(
      get_bc_hom_step,
      0.01,
      0, scenario_record.max_step, max_interval=512,
      err_func=err_bc_hom,
  )

  return x_bc_hom, bc_hom_smpl


_, state_vars, cycles = c.get_dep_graph()

valid_state_vars = {
    'scenario_record',
    'active_threshold',
    'min_inactive_value'
}

invalid_state_vars = [x for x in state_vars if x not in valid_state_vars]

if invalid_state_vars:
  raise ValueError(
      'Invalid state variables detected: ' +
      ','.join([str(c) for c in invalid_state_vars])
  )

if cycles:
  raise ValueError(
      'Cyclic dependencies detected: ' +
      ','.join([str(c) for c in cycles])
  )
