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


def plot_heatmap(
    peaks: int | None = None,
    f: Callable[[ScenarioStatistics],
                float] = lambda x: x.grad_index,  # type: ignore
    stats_name='grad. index',

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
      n_row=1, n_col=3, hw_ratio=1
  )

  ax1, ax2, ax3 = axes

  im1 = ax1.imshow(grad_index_mean_st, cmap='YlGnBu', vmin=0, vmax=heatmap_max)
  ax1.set_title(f'(a) {stats_name} (structure)', loc='left')
  fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

  # 在ax1上添加标准差圆圈
  for i in range(grad_index_std_st.shape[0]):
    for j in range(grad_index_std_st.shape[1]):
      std_val = grad_index_std_st[i, j]
      if not np.isnan(std_val):
        ax1.scatter(j, i, s=std_val * 1000, facecolors='none',
                    edgecolors='dimgray', linewidths=2, alpha=0.6)

  im2 = ax2.imshow(grad_index_mean_op, cmap='YlGnBu', vmin=0, vmax=heatmap_max)
  ax2.set_title(f'(b) {stats_name} (opinion)', loc='left')
  fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

  # 在ax2上添加标准差圆圈
  for i in range(grad_index_std_op.shape[0]):
    for j in range(grad_index_std_op.shape[1]):
      std_val = grad_index_std_op[i, j]
      if not np.isnan(std_val):
        ax2.scatter(j, i, s=std_val * 1000, facecolors='none',
                    edgecolors='dimgray', linewidths=2, alpha=0.6)

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


if __name__ == '__main__':
  def f_grad(x: ScenarioStatistics) -> float:
    assert x.grad_index is not None
    return x.grad_index

  def f_p_backdrop(x: ScenarioStatistics) -> float:
    assert x.p_backdrop is not None
    return x.p_backdrop

  plt_save_and_close(
    plot_heatmap(
      peaks=None, f=f_grad, stats_name='grad. index'
    ), 'fig/f_grad_index'
  )

  plt_save_and_close(
    plot_heatmap(
      peaks=1, f=f_p_backdrop,
               stats_name='PBD, #P=1', heatmap_max=2, diff_range=1,
    ), 'fig/f_p_backdrop_1_peak'
  )
  
  # plot_heatmap(peaks=2, f=f_p_backdrop,
  #              stats_name='PBD, #P=2', heatmap_max=2).show()
