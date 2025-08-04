from typing import Dict, List, Iterable, Callable, TypeVar, Tuple
from numpy.typing import NDArray

import os
import copy

import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.plot import plt_figure, plt_save_and_close
from utils.sqlalchemy import create_db_engine_and_session
import works.config as cfg
from works.stat.types import ScenarioStatistics


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

# parameters

plot_path = cfg.SIMULATION_STAT_DIR
stats_db_path = os.path.join(plot_path, 'stats.db')

rs_keys = list(cfg.rs_names)
tw_keys = cfg.retweet_rate_array.tolist()


plt.rcParams.update({'font.size': 18})

# utilities

# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


bc_inst_orig: List[List[List[ScenarioStatistics]]] = []
rw_rev_dict: Dict[float, int] = {}
dc_rev_dict: Dict[float, int] = {}
for i, rw in enumerate(cfg.rewiring_rate_array):
  rw_rev_dict[rw] = i
  lst = []
  for j, dc in enumerate(cfg.decay_rate_array):
    dc_rev_dict[dc] = j
    lst.append([])
  bc_inst_orig.append(lst)


def _create_evaluator(
    full_data_selector: Iterable[ScenarioStatistics]
):
  bc_inst = copy.deepcopy(bc_inst_orig)
  for datum in tqdm(full_data_selector, bar_format=short_progress_bar):
    if datum is None:
      continue
    rw_index = rw_rev_dict[datum.rewiring]
    dc_index = dc_rev_dict[datum.decay]
    bc_inst[rw_index][dc_index].append(datum)

  def eval_func(
      f_ext: Callable[[ScenarioStatistics], float]
  ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    bc_inst_num: List[List[List[float]]] = []
    for l_rw in bc_inst:
      bc_inst_num_row = []
      for l_dc_rw in l_rw:
        bc_inst_num_cell: List[float] = [f_ext(x) for x in l_dc_rw]
        bc_inst_num_row.append([
            np.mean(bc_inst_num_cell) if bc_inst_num_cell else None,
            np.std(bc_inst_num_cell) if bc_inst_num_cell else None,
        ])
      bc_inst_num.append(bc_inst_num_row)

    # (rewiring, influence, [mean, std])
    bc_arr = np.array(bc_inst_num, dtype=float)
    bc_mean, bc_std = bc_arr[:, :, 0], bc_arr[:, :, 1]
    return bc_mean, bc_std

  return eval_func


def create_heatmap_evaluator(
    session: Session,
    rs_key: str,
    opinion_peak_count: int | None = None,
):
  full_data_selector: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      *((ScenarioStatistics.last_opinion_peak_count == opinion_peak_count,)
        if opinion_peak_count is not None else ()),
      ScenarioStatistics.name.startswith('s_grad'),
      ScenarioStatistics.recsys_type == rs_key,
      ScenarioStatistics.grad_index.is_not(None),
  )
  return _create_evaluator(full_data_selector)


def create_heatmap_evaluator_retweet(
    session: Session,
    retweet_rate: float,
    rs_key: str | None = None,
):
  full_data_selector: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.name.startswith('s_grad'),
      *((ScenarioStatistics.recsys_type == rs_key,) if rs_key else ()),
      ScenarioStatistics.retweet == retweet_rate,
      ScenarioStatistics.grad_index.is_not(None),
  )
  return _create_evaluator(full_data_selector)


def plot_heatmap_by_rec(
    peaks: int | None = None,
    f: Callable[[ScenarioStatistics],
                float] = lambda x: x.grad_index,  # type: ignore
    stats_name='grad. index',
    with_diff=True,
    with_std=True,
    cmap='RdYlBu',

    heatmap_min=0,
    heatmap_max=1,

    diff_range=0.6,
):
  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base)
  evaluator_st = create_heatmap_evaluator(session, "StructureM9", peaks)
  evaluator_op = create_heatmap_evaluator(session, "OpinionM9", peaks)

  grad_index_mean_st, grad_index_std_st = evaluator_st(f)
  grad_index_mean_op, grad_index_std_op = evaluator_op(f)

  grad_index_diff = grad_index_mean_op - grad_index_mean_st

  fig, axes = plt_figure(
      n_row=1,
      n_col=3 if with_diff else 2,
      hw_ratio=1,
      total_width=18 if with_diff else 12,
  )

  if with_diff:
    ax1, ax2, ax3 = axes
  else:
    ax1, ax2 = axes

  im1 = ax1.imshow(grad_index_mean_st, cmap=cmap,
                   vmin=heatmap_min, vmax=heatmap_max)
  ax1.set_title(f'(a) {stats_name} (structure)', loc='left')
  fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

  # 在ax1上添加标准差圆圈
  for i in range(grad_index_std_st.shape[0]) if with_std else ():
    for j in range(grad_index_std_st.shape[1]):
      std_val = grad_index_std_st[i, j]
      if not np.isnan(std_val):
        ax1.scatter(j, i, s=std_val * 1000, facecolors='none',
                    edgecolors='midnightblue', linewidths=2, alpha=0.6)

  im2 = ax2.imshow(grad_index_mean_op, cmap=cmap,
                   vmin=heatmap_min, vmax=heatmap_max)
  ax2.set_title(f'(b) {stats_name} (opinion)', loc='left')
  fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

  # 在ax2上添加标准差圆圈
  for i in range(grad_index_std_op.shape[0]) if with_std else ():
    for j in range(grad_index_std_op.shape[1]):
      std_val = grad_index_std_op[i, j]
      if not np.isnan(std_val):
        ax2.scatter(j, i, s=std_val * 1000, facecolors='none',
                    edgecolors='midnightblue', linewidths=2, alpha=0.6)

  if with_diff:
    im3 = ax3.imshow(grad_index_diff, cmap='RdBu',
                     vmin=-diff_range, vmax=diff_range)
    ax3.set_title('(c) difference', loc='left')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

  for ax in axes:
    ax.invert_yaxis()
    ax.set_xticks(np.arange(cfg.decay_rate_array.size))
    ax.set_yticks(np.arange(cfg.rewiring_rate_array.size))

    ax.set_xlabel('influence')
    ax.set_xticklabels(cfg.decay_rate_array, rotation=90)

    if ax is ax1:
      ax.set_ylabel('rewiring')
      ax.set_yticklabels(cfg.rewiring_rate_array)
    else:
      ax.set_ylabel('')
      ax.set_yticklabels([])

  fig.tight_layout()

  session.close()
  engine.dispose()

  return fig


def plot_heatmap_by_retweet(
    f: Callable[[ScenarioStatistics],
                float] = lambda x: x.grad_index,  # type: ignore
    stats_name='grad. index',
    heatmap_min=0,
    heatmap_max=1,
):
  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base)

  evaluators = [
      create_heatmap_evaluator_retweet(session, r, k)
      for r, k in [(0, 'StructureM9'), (0, 'OpinionM9'), (0.5, 'StructureM9'), (0.5, 'OpinionM9'), ]
  ]

  labels = [
      'St., p=0',
      'St., p=0',
      'Op., p=0.5',
      'Op., p=0.5',
  ]

  stats = [e(f) for e in evaluators]

  fig, axes = plt_figure(
      n_row=1, n_col=4, hw_ratio=1, total_width=24,
  )

  for i, ax in enumerate(axes):

    grad_index_mean_st, grad_index_std_st = stats[i]

    im1 = ax.imshow(grad_index_mean_st, cmap='YlGnBu',
                    vmin=heatmap_min, vmax=heatmap_max)
    ax.set_title(f'(a) {stats_name} ({labels[i]})', loc='left')
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # 在ax1上添加标准差圆圈
    # for i in range(grad_index_std_st.shape[0]):
    #   for j in range(grad_index_std_st.shape[1]):
    #     std_val = grad_index_std_st[i, j]
    #     if not np.isnan(std_val):
    #       ax.scatter(j, i, s=std_val * 1000, facecolors='none',
    #                   edgecolors='midnightblue', linewidths=2, alpha=0.6)

  for ax in axes:
    ax.invert_yaxis()
    ax.set_xticks(np.arange(cfg.decay_rate_array.size))
    ax.set_yticks(np.arange(cfg.rewiring_rate_array.size))

    ax.set_xlabel('influence')
    ax.set_xticklabels(cfg.decay_rate_array, rotation=90)

    if ax is ax:
      ax.set_ylabel('rewiring')
      ax.set_yticklabels(cfg.rewiring_rate_array)
    else:
      ax.set_ylabel('')
      ax.set_yticklabels([])

  fig.tight_layout()

  session.close()
  engine.dispose()

  return fig


if __name__ == '__main__':
  def f_grad(x: ScenarioStatistics) -> float:
    assert x.grad_index is not None
    return x.grad_index

  def f_p_backdrop(x: ScenarioStatistics) -> float:
    assert x.p_backdrop is not None and x.p_index is not None
    return x.p_backdrop / x.p_index[-1]

  def f_active_step(x: ScenarioStatistics) -> float:
    assert x.active_step is not None
    return np.log10(x.active_step)

  plt_save_and_close(
      plot_heatmap_by_rec(
          peaks=None, f=f_grad, stats_name='grad. index'
      ), 'fig/f_grad_index'
  )

  plt_save_and_close(
      plot_heatmap_by_rec(
          peaks=1, f=f_p_backdrop,
          stats_name='MPI, #P=1', heatmap_min=1, heatmap_max=8, diff_range=1,
          with_diff=False, cmap='YlGnBu', with_std=False,
      ), 'fig/f_p_backdrop_1_peak'
  )

  plt_save_and_close(
      plot_heatmap_by_retweet(
          f=f_active_step,
          stats_name='log10(t_a)', heatmap_max=5, heatmap_min=1,
      ), 'fig/f_p_active_step'
  )

  # plot_heatmap(peaks=2, f=f_p_backdrop,
  #              stats_name='PBD, #P=2', heatmap_max=2).show()
