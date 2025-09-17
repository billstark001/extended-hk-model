

from result_interp.record import RawSimulationRecord
from utils.plot import plt_figure
import works.config as cfg
from works.stat.context import c
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# key = 's_grad_sim1_rw7_dc0_rt3_op6'

# key = 's_grad_sim9_rw3_dc7_rt1_op6'
# key = 's_grad_sim9_rw7_dc0_rt3_op6'
# key = 's_grad_sim9_rw2_dc5_rt3_op6'

# key = 's_grad_sim8_rw0_dc2_rt1_op6'
# key = 's_grad_sim8_rw0_dc0_rt1_op2'

def get_rec_by_key(key: str) -> RawSimulationRecord:
  rc1 = [x for x in cfg.all_scenarios_grad if x['UniqueName'] == key][0]

  rec = RawSimulationRecord(
      cfg.get_workspace_dir(name='hdd_1t_raw'), rc1,
  )
  return rec


def seq_to_map(seq: np.ndarray) -> np.ndarray:
  map_raw = np.concatenate([
      seq[:-1, :], seq[1:, :],
  ])
  map_flattened = map_raw.reshape(2, -1).T
  return map_flattened


def k_points_to_map(k_points: np.ndarray, max_val=15) -> np.ndarray:
  # k_points: (*, 2)
  cnt_mat = np.zeros((max_val + 1, max_val + 1), dtype=int)

  for x in range(max_val + 1):
    for y in range(max_val + 1):
      cnt_mat[y, x] = np.sum((k_points[:, 0] == x) & (k_points[:, 1] == y))

  return cnt_mat


def eval_rec(rec: RawSimulationRecord, key: str):

  c.set_state(
      active_threshold=0.98,
      min_inactive_value=0.75,
      scenario_record=rec,
  )

  active_step: float = c.active_step

  t_seq = np.arange(rec.opinions.shape[0]) / active_step

  print(key, rec.max_step)

  c.debug = True

  k_seq = rec.agent_numbers[:, :, 0]
  x_seq = rec.opinions[:, :]
  k_seq_l_1 = np.copy(k_seq)
  k_seq_l_1[k_seq_l_1 < 1] = 1
  dx_n_seq = rec.agent_opinion_sums[:, :, 0] / k_seq_l_1
  dx_n_map_raw = np.concatenate([
      x_seq, dx_n_seq,
  ])
  dx_n_map_trans = dx_n_map_raw.reshape(2, -1).T
  dx_n_points = 50000
  if dx_n_map_trans.shape[0] > dx_n_points:
    indices = np.random.choice(
        dx_n_map_trans.shape[0], size=dx_n_points, replace=False)
    dx_n_map = dx_n_map_trans[indices]
  else:
    dx_n_map = dx_n_map_trans

  k_points = seq_to_map(k_seq)
  k_points[k_points > 15] = 15
  k_map_raw = k_points_to_map(k_points, max_val=15)
  k_map = np.log1p(k_map_raw)

  x_seq_resampled = interp1d(
      t_seq, x_seq, axis=0, kind='linear', fill_value='extrapolate'  # type: ignore
  )(np.linspace(0, 1, num=50))

  x_map = seq_to_map(x_seq_resampled)

  fig, axes = plt_figure(n_row=1, n_col=3, total_width=12, hw_ratio=1)

  ax_x, ax_dx, ax_k = axes
  ax_x.scatter(x_map[:, 0], x_map[:, 1], s=0.05, alpha=0.5)
  ax_dx.scatter(dx_n_map[:, 0], dx_n_map[:, 1], s=0.05, alpha=0.5)
  ax_k.imshow(k_map, origin='lower', cmap='YlGnBu')

  ax_x.set_xlim(-1, 1)
  ax_x.set_ylim(-1, 1)
  ax_dx.set_xlim(-1, 1)
  ax_dx.set_ylim(-1, 1)

  ax_x.grid(True, linestyle='--', alpha=0.5)
  ax_dx.grid(True, linestyle='--', alpha=0.5)
  
  ax_x.set_title(key)
  ax_x.set_xlabel(r'$x_i(t)$')
  ax_x.set_ylabel(r'$x_i(t + dt)$')
  
  ax_dx.set_xlabel(r'$x_i(t)$')
  ax_dx.set_ylabel(r'$\Delta_N x_i(t)$')
  
  ax_k.set_xlabel(r'$k_i(t)$')
  ax_k.set_ylabel(r'$k_i(t + 1)$')
  fig.colorbar(ax_k.images[0], ax=ax_k, label=r'$log(1 + \#)$')

  plt.show()

if __name__ == '__main__':
  # keys = [
  #     's_grad_sim9_rw2_dc7_rt0_st',
  #     's_grad_sim9_rw2_dc7_rt3_st',
  #     's_grad_sim9_rw2_dc7_rt0_op6',
  #     's_grad_sim9_rw2_dc_rt3_op6',

  #     's_grad_sim9_rw7_dc0_rt0_st',
  #     's_grad_sim9_rw7_dc0_rt3_st',
  #     's_grad_sim9_rw7_dc0_rt0_op6',
  #     's_grad_sim9_rw7_dc0_rt3_op6',
  # ]
  keys = [
      f's_grad_sim9_rw3_dc{i}_rt0_op6'
      for i in range(7, -1, -1)
  ]
  for key in keys:
    rec = get_rec_by_key(key)
    with rec:
      eval_rec(rec, key)