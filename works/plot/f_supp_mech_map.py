from typing import Tuple, TypeAlias, List

from result_interp.record import RawSimulationRecord
from utils.plot import plt_figure, plt_save_and_close, setup_paper_params
import works.config as cfg
from works.stat.context import c
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import interp1d


def seq_to_map(seq: np.ndarray, diff=False) -> np.ndarray:
  # agent_num = seq.shape[1]
  map_raw = np.array([
      seq[:-1, :], (seq[1:, :] - seq[:-1, :]) if diff else seq[1:, :],
  ])  # (2, t-1, n)
  map_flattened = map_raw.transpose((2, 1, 0))  # (n, t-1, 2)
  return map_flattened


def k_points_to_map(k_points: np.ndarray, max_val=15) -> np.ndarray:
  # k_points: (*, 2)
  cnt_mat = np.zeros((max_val + 1, max_val + 1), dtype=int)

  for x in range(max_val + 1):
    for y in range(max_val + 1):
      cnt_mat[y, x] = np.sum((k_points[:, 0] == x) & (k_points[:, 1] == y))

  return cnt_mat


ParsedRecord: TypeAlias = 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]'


def get_basic_data(rec: RawSimulationRecord) -> ParsedRecord:

  if globals().get('basic_data_cache', None) is None:
    globals()['basic_data_cache'] = {}
  rec_key = rec.unique_name
  if rec_key in globals()['basic_data_cache']:
    return globals()['basic_data_cache'][rec_key]

  c.set_state(
      active_threshold=0.98,
      min_inactive_value=0.75,
      scenario_record=rec,
  )
  c.debug = True

  active_step: float = c.active_step

  t_seq = np.arange(rec.opinions.shape[0]) / active_step

  print(rec.max_step)

  k_seq = rec.agent_numbers[:, :, 0]
  x_seq = rec.opinions[:, :]
  k_seq_l_1 = np.copy(k_seq)
  k_seq_l_1[k_seq_l_1 < 1] = 1
  dx_n_seq = rec.agent_opinion_sums[:, :, 0] / k_seq_l_1

  ret: ParsedRecord = (t_seq, x_seq, dx_n_seq, k_seq)
  globals()['basic_data_cache'][rec_key] = ret
  return ret


plot_res = 50
cmap_name = 'managua'


def plot_colorbar(fig: Figure, axes: List[Axes]):
  norm = mpl.colors.Normalize(vmin=0, vmax=1)  # type: ignore
  sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
  sm.set_array([])  # only needed for older matplotlib versions
  cbar = fig.colorbar(sm, ax=axes, orientation='vertical', aspect=30, pad=0.01)
  cbar.set_label('$t_n$')
  return cbar


def eval_rec_k_map(ax_k: Axes, rec_parsed: ParsedRecord):

  _, _, _, k_seq = rec_parsed

  k_seq = rec.agent_numbers[:, :, 0]
  k_seq_l_1 = np.copy(k_seq)
  k_seq_l_1[k_seq_l_1 < 1] = 1

  k_points = seq_to_map(k_seq).reshape(-1, 2)  # (n*(t-1), 2)
  k_points[k_points > 15] = 15
  k_map_raw = k_points_to_map(k_points, max_val=15)
  k_map = np.log1p(k_map_raw)

  ax_k.imshow(k_map, origin='lower', cmap='YlGnBu')


extrapolate_fill_value: float = 'extrapolate'  # type: ignore


def eval_rec_x_map(ax_x: Axes, rec_parsed: ParsedRecord):

  t_seq, x_seq, _, _ = rec_parsed
  xx_resample = np.linspace(0, x_seq.shape[0] - 1, num=plot_res)

  x_seq_resampled = interp1d(
      np.arange(x_seq.shape[0]), x_seq, axis=0, kind='linear', fill_value=extrapolate_fill_value
  )(xx_resample)
  t_seq_resampled = interp1d(
      np.arange(t_seq.shape[0]), t_seq, kind='linear', fill_value=extrapolate_fill_value
  )(xx_resample)

  x_map = seq_to_map(x_seq_resampled, diff=True)
  cmap = plt.get_cmap(cmap_name)

  for time_idx in range(x_map.shape[1] - 1):
    color = cmap(t_seq_resampled[time_idx])
    ax_x.plot(
        x_map[:, time_idx:time_idx+2, 0].T,
        x_map[:, time_idx:time_idx+2, 1].T,
        color=color, linewidth=0.5, alpha=min(0.1 + t_seq_resampled[time_idx] * 0.5, 1)
    )


def eval_rec_dx_map(ax_dx: Axes, rec_parsed: ParsedRecord):

  t_seq, x_seq, dx_n_seq, _ = rec_parsed
  xx_resample = np.linspace(0, x_seq.shape[0] - 1, num=plot_res)

  x_seq_resampled = interp1d(
      np.arange(x_seq.shape[0]), x_seq, axis=0, kind='linear', fill_value=extrapolate_fill_value
  )(xx_resample)
  t_seq_resampled = interp1d(
      np.arange(t_seq.shape[0]), t_seq, kind='linear', fill_value=extrapolate_fill_value
  )(xx_resample)
  dx_n_seq_resampled = interp1d(
      np.arange(dx_n_seq.shape[0]), dx_n_seq, axis=0, kind='linear', fill_value=extrapolate_fill_value
  )(xx_resample)

  dx_n_map_raw = np.array([
      x_seq_resampled[:-1], dx_n_seq_resampled[1:],
  ])  # (2, t, n)
  dx_n_map_trans = dx_n_map_raw.transpose((2, 1, 0))
  cmap = plt.get_cmap(cmap_name)

  for time_idx in range(dx_n_map_trans.shape[1] - 1):
    color = cmap(t_seq_resampled[time_idx])
    ax_dx.plot(
        dx_n_map_trans[:, time_idx:time_idx+2, 0].T,
        dx_n_map_trans[:, time_idx:time_idx+2, 1].T,
        color=color, linewidth=0.5, alpha=min(0.1 + t_seq_resampled[time_idx] * 0.5, 1)
    )


def set_ax_format(ax: Axes, dt: float | str = 0.02, ylim=0.5, xlabel=True, ylabel=True):
  ax.set_xlim(-1, 1)
  ax.set_ylim(-ylim, ylim)
  ax.grid(True, linestyle='--', alpha=0.5)
  if xlabel:
    ax.set_xlabel(r'$x_i(t)$')
  else:
    ax.set_xticklabels([])
  if ylabel:
    ax.set_ylabel(rf'$\Delta_{{{dt}}} x_i(t)$')
  else:
    ax.set_yticklabels([])
  return ax


if __name__ == '__main__':

  setup_paper_params()

  recs = [
      RawSimulationRecord(
          cfg.get_workspace_dir(name='local_mech'), d,
      ) for d in cfg.all_scenarios_mech
  ]

  recs_g1 = recs[:4]
  recs_g2 = recs[4:]

  # group 1
  fig, (axes_r1, axes_r2) = plt_figure(n_row=2, n_col=4)

  for i, rec in enumerate(recs_g1):
    set_ax_format(axes_r1[i], xlabel=False, ylabel=(i == 0), dt=1 / plot_res)
    set_ax_format(axes_r2[i], xlabel=True, ylabel=(i == 0), dt='N')
    with rec:
      rec_parsed = get_basic_data(rec)
    eval_rec_x_map(axes_r1[i], rec_parsed)
    eval_rec_dx_map(axes_r2[i], rec_parsed)

  for i, title in enumerate(['baseline', '+influence', '+retweet', 'opinion rec.']):
    char = chr(ord('a') + i)
    axes_r1[i].set_title(f'({char}1) {title}', loc='left')
    axes_r2[i].set_title(f'({char}2)', loc='left')

  plot_colorbar(fig, [*axes_r1, *axes_r2])

  # fig.tight_layout()
  plt_save_and_close(fig, 'fig/f_supp_mech_map_g1')

  del fig, axes_r1, axes_r2

  # group 2
  fig, (axes_r1, axes_r2) = plt_figure(n_row=2, n_col=5, total_width=14)

  for i, rec in enumerate(recs_g2):
    set_ax_format(axes_r1[i], xlabel=False, ylabel=(i == 0), dt=1 / plot_res)
    set_ax_format(axes_r2[i], xlabel=True, ylabel=(i == 0), dt='N')
    with rec:
      rec_parsed = get_basic_data(rec)
    eval_rec_x_map(axes_r1[i], rec_parsed)
    eval_rec_dx_map(axes_r2[i], rec_parsed)

  for i in range(5):
    char = chr(ord('a') + i)
    axes_r1[i].set_title(f'({char}1)', loc='left')
    axes_r2[i].set_title(f'({char}2)', loc='left')

  plot_colorbar(fig, [*axes_r1, *axes_r2])

  # fig.tight_layout()
  plt_save_and_close(fig, 'fig/f_supp_mech_map_g2')
