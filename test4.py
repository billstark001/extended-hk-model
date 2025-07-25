import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from stats.distance_c import DistanceCollectorContinuous
from utils.stat import adaptive_discrete_sampling, merge_data_with_axes
import works.config as c
from result_interp import RawSimulationRecord


def err_func(x1, x2, x3, t3):
  return max(
      abs(x1[i] * (1 - t3) + x2[i] * t3 - x3[i])
      for i in range(4)
  )


dis = DistanceCollectorContinuous(
    use_js_divergence=True,
    min_bandwidth=0.01,
)

rec = RawSimulationRecord(
    c.get_workspace_dir(),
    c.all_scenarios_grad[10]
)


def calc_distance(step: int):
  #   print('dis', step)
  graph = rec.get_graph(step)
  opinion = rec.opinions[step]

  dis_res = dis.collect('d', graph, opinion)

  d_rand_o = dis_res['d-rand-o']
  d_rand_s = dis_res['d-rand-s']
  d_worst_o = 1 - dis_res['d-worst-o']
  d_worst_s = 1 - dis_res['d-worst-s']

  return d_rand_o, d_rand_s, d_worst_o, d_worst_s


def calc_homophily(step: int):
  f = rec.followers.astype(float)
  f_slice = rec.agent_numbers[step, :, 0].astype(float)
  return np.mean(f_slice / f)


with rec:

  cx, cy = adaptive_discrete_sampling(
      calc_distance,
      0.01,
      0,
      rec.max_step,
      max_interval=512,
      err_func=err_func,
  )

  cx = np.array(cx)
  cy = np.array(cy)
  cy = np.clip(cy, 0, 1)

  cxh, cyh = adaptive_discrete_sampling(
      calc_homophily,
      0.01,
      0, rec.max_step, max_interval=512,
  )

  x, (y_dist, y_homo) = merge_data_with_axes(
      (cx, cy),
      (cxh, cyh)
  )
