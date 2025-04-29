import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from stats.distance_c import DistanceCollectorContinuous
from utils.stat import adaptive_discrete_sampling
import works.config as c
from result_interp import RawSimulationRecord


dis = DistanceCollectorContinuous(
    use_js_divergence=True
)

rec = RawSimulationRecord(
    c.SIMULATION_RESULT_DIR,
    c.all_scenarios_grad[8]
)
rec.load()


def err_func(x1, x2, x3, t3): return max(
    abs(x1[i] * (1 - t3) + x2[i] * t3 - x3[i])
    for i in range(4)
)


def calc_distance(step: int):
  print('dis', step)
  graph = rec.get_graph(step)
  opinion = rec.opinions[step]

  dis_res = dis.collect('d', graph, opinion)

  d_rand_o = dis_res['d-rand-o']
  d_rand_s = dis_res['d-rand-s']
  d_worst_o = dis_res['d-worst-o']
  d_worst_s = dis_res['d-worst-s']

  return d_rand_o, d_rand_s, d_worst_o, d_worst_s


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

def calc_homophily(step: int):
  f = rec.followers.astype(float)
  f_slice = rec.agent_numbers[step, :, 0].astype(float)
  return np.mean(f_slice / f)


cxh, cyh = adaptive_discrete_sampling(
    calc_homophily, 0.01,
    0, rec.max_step,
    max_interval=512,
)
