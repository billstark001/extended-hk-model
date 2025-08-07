from typing import Dict, Tuple, List

from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from utils.plot import plt_figure
from result_interp.parse_events_db import RewiringEventBody, batch_load_event_bodies, get_events_by_step_range
from result_interp.record import RawSimulationRecord
import works.config as cfg
from works.stat.context import c


def optimized_bidirectional_bfs(G: nx.DiGraph, u: int, v: int, n: int | None = None) -> int:
  # n: 节点总数
  if u == v:
    return 0
  if n is None:
    n = len(G.nodes)
  visited_u = np.zeros(n, dtype=bool)
  visited_v = np.zeros(n, dtype=bool)
  dist_u = np.full(n, -1, dtype=np.int16)
  dist_v = np.full(n, -1, dtype=np.int16)

  queue_u = deque([u])
  queue_v = deque([v])
  visited_u[u] = True
  visited_v[v] = True
  dist_u[u] = 0
  dist_v[v] = 0

  while queue_u and queue_v:
    # always expand smaller frontier
    if len(queue_u) <= len(queue_v):
      for _ in range(len(queue_u)):
        curr = queue_u.popleft()
        for nei in G.successors(curr):
          if not visited_u[nei]:
            visited_u[nei] = True
            dist_u[nei] = dist_u[curr] + 1
            if visited_v[nei]:
              return dist_u[nei] + dist_v[nei] # type: ignore
            queue_u.append(nei)
    else:
      for _ in range(len(queue_v)):
        curr = queue_v.popleft()
        for nei in G.predecessors(curr):  # 反向扩展
          if not visited_v[nei]:
            visited_v[nei] = True
            dist_v[nei] = dist_v[curr] + 1
            if visited_u[nei]:
              return dist_u[nei] + dist_v[nei] # type: ignore
            queue_v.append(nei)
  return -1  # not reachable

# key = 's_grad_sim1_rw7_dc0_rt3_op6'


key = 's_grad_sim8_rw7_dc0_rt0_op6'

# key = 's_grad_sim8_rw0_dc2_rt1_op6'
# key = 's_grad_sim8_rw0_dc0_rt1_op2'

rc1 = [x for x in cfg.all_scenarios_grad if x['UniqueName'] == key][0]

rec = RawSimulationRecord(
    cfg.get_workspace_dir(name='ssd_ug_m2'), rc1,
)

print('Loading...')

with rec:

  print('Loaded.')

  c.set_state(
      active_threshold=0.98,
      min_inactive_value=0.75,
      scenario_record=rec,
  )

  # opinion diff

  print('opinion diff')

  active_step = float(c.active_step)
  opinions_diff_abs = np.abs(rec.opinions[1:] - rec.opinions[:-1])

  step_mask = np.arange(1, opinions_diff_abs.shape[0] + 1, 1, dtype=int)

  points = []
  seg = 0.2
  x_seg = np.arange(0, 1, seg)
  for seg_start in x_seg:
    seg_end = seg_start + seg
    step_mask_seg = np.logical_and(
        step_mask >= seg_start * active_step, step_mask < seg_end * active_step)
    sum_seg = np.sum(opinions_diff_abs[step_mask_seg], axis=0)
    mean_seg = np.mean(sum_seg)
    std_seg = np.std(sum_seg)
    points.append((mean_seg, std_seg))

  final_mean, final_std = np.array(points).T

  # plot
  plt.bar([str(x) for x in x_seg], final_mean, yerr=final_std)
  # for i, v in enumerate(final_mean):
  #     plt.text(i, float(v + 0.5), str(v), ha='center', va='bottom')
  plt.title(key)
  plt.grid(True)
  # plt.ylim(0, 0.4)
  plt.show()
  
  # distance

  print('distance')
  
  all_events = get_events_by_step_range(rec.events_db, 0, rec.max_step + 1, 'Rewiring')
  all_events_body_loaded = batch_load_event_bodies(rec.events_db, all_events)
  ev_dict_unfollow: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
  ev_dict_follow: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
  
  for e in all_events_body_loaded:
    body: RewiringEventBody = e.body # type: ignore
    ev_dict_unfollow[e.step].append((e.agent_id, body.unfollow))
    ev_dict_follow[e.step - 1].append((e.agent_id, body.follow))
  
  all_steps = list(set(ev_dict_unfollow.keys()).union(ev_dict_follow.keys()))
  all_steps.sort()
  
  path_length_unfollow: List[Tuple[int, int]] = []
  path_length_follow: List[Tuple[int, int]] = []
  
  for step in tqdm(all_steps):
    pairs_unfollow = ev_dict_unfollow[step]
    pairs_follow = ev_dict_follow[step]
    
    G = rec.get_graph(step)
    
    for u, v in pairs_unfollow:
      path_length_unfollow.append((step, optimized_bidirectional_bfs(G, u, v)))
    for u, v in pairs_follow:
      path_length_follow.append((step, optimized_bidirectional_bfs(G, u, v)))
      
  path_length_unfollow_arr = np.array(path_length_unfollow, dtype=float).T
  path_length_follow_arr = np.array(path_length_follow, dtype=float).T
  
  uf_length, uf_counts = np.unique(path_length_unfollow_arr[1], return_counts=True)
  fl_length, fl_counts = np.unique(path_length_follow_arr[1], return_counts=True)
  
  fig, (ax1, ax2) = plt_figure(n_row=1, n_col=2, total_width=12)

  ax1.bar(uf_length, uf_counts)
  ax1.set_yscale('log')
  ax1.grid(True)

  ax2.bar(fl_length, fl_counts)
  ax2.set_yscale('log')
  ax2.grid(True)
  
  ax1.set_title(key, loc='left')
  _pf = path_length_unfollow_arr[1]
  ax1.set_xlabel(f'unfollow, avg: {np.mean(_pf[_pf > 0]):.4f}')
  _pf = path_length_follow_arr[1]
  ax2.set_xlabel(f'follow, avg: {np.mean(_pf[_pf > 0]):.4f}')
  
  fig.show()
