import matplotlib.pyplot as plt


from stats.distance_c import DistanceCollectorContinuous
from utils.stat import adaptive_discrete_sampling
import works.config as c
from result_interp import RawSimulationRecord


dis = DistanceCollectorContinuous(
  use_js_divergence=True
)

rec = RawSimulationRecord(
  c.SIMULATION_RESULT_DIR, 
  c.all_scenarios_grad[5]
)
rec.load()

err_func = lambda x1, x2: max(x1[i] - x2[i] for i in range(4))

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

curves = adaptive_discrete_sampling(
  calc_distance,
  0.01,
  0,
  rec.max_step,
  max_interval=100,
  err_func=err_func,
)
