

from result_interp.record import RawSimulationRecord
from utils.plot import plt_figure
import works.config as cfg
from works.stat.context import c
import numpy as np
import matplotlib.pyplot as plt

# key = 's_grad_sim1_rw7_dc0_rt3_op6'

key = 's_grad_sim9_rw7_dc0_rt1_op6'

# key = 's_grad_sim8_rw0_dc2_rt1_op6'
# key = 's_grad_sim8_rw0_dc0_rt1_op2'

rc1 = [x for x in cfg.all_scenarios_grad if x['UniqueName'] == key][0]

rec = RawSimulationRecord(
  cfg.get_workspace_dir(default='ssd_tmp'), rc1,
)

def seq_to_map(seq: np.ndarray) -> np.ndarray:
  map_raw = np.concatenate([
    seq[1:, :], seq[:-1, :],
  ])
  map_flattened = map_raw.reshape(2, -1).T
  return map_flattened

def k_points_to_map(k_points: np.ndarray, max_val = 15) -> np.ndarray:
  # k_points: (*, 2)
  cnt_mat = np.zeros((max_val + 1, max_val + 1), dtype=int)
  
  for x in range(max_val + 1):
    for y in range(max_val + 1):
      cnt_mat[x, y] = np.sum((k_points[:, 0] == x) & (k_points[:, 1] == y))

  return cnt_mat

with rec:

  c.set_state(
      active_threshold=0.98,
      min_inactive_value=0.75,
      scenario_record=rec,
  )
  
  print(rec.max_step)
  
  c.debug = True
  
  
  k_seq = rec.agent_numbers[:, :, 0]
  x_seq = rec.opinions[:, :]
  dx_seq = rec.agent_opinion_sums[:, :, 0] / k_seq

  dx_map = seq_to_map(dx_seq)
  x_map = seq_to_map(x_seq)
  k_points = seq_to_map(k_seq)
  k_points[k_points > 15] = 15
  k_map = k_points_to_map(k_points, max_val=15)

  fig, axes = plt_figure(n_row=1, n_col=3, total_width=24)
  
  ax_x, ax_dx, ax_k = axes
  ax_x.scatter(x_map[:, 0], x_map[:, 1], s=0.01, alpha=0.1)
  ax_dx.scatter(dx_map[:, 0], dx_map[:, 1], s=0.01, alpha=0.1)
  ax_k.imshow(k_map, origin='lower', cmap='YlGnBu')
  
  ax_x.set_xlim(-1, 1)
  ax_x.set_ylim(-1, 1)
  ax_dx.set_xlim(-1, 1)
  ax_dx.set_ylim(-1, 1)
  
  for ax in axes:
    ax.grid(True)
    
  fig.show()