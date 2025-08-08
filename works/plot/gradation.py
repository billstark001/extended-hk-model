from typing import Dict, List, Iterable, Callable, TypeVar, Tuple

import os
import copy
import random

import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from utils.plot import plt_figure, get_colormap
from utils.sqlalchemy import create_db_session
import works.config as cfg
from works.plot.data_utils import piecewise_linear_integral_trapz
from works.stat.types import ScenarioStatistics
from matplotlib.colors import LinearSegmentedColormap


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


def create_heatmap_evaluator_raw(
    session: Session,
    rs_name: str,
    retweet_rate: float,
):
  rs_key, rs_retain = cfg.rs_names[rs_name]
  full_data_selector: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.last_opinion_peak_count == 3,
      ScenarioStatistics.name.startswith('s_grad'),
      ScenarioStatistics.recsys_type == rs_key,
      ScenarioStatistics.tweet_retain_count == rs_retain,
      ScenarioStatistics.retweet == retweet_rate,
  )
  bc_inst = copy.deepcopy(bc_inst_orig)
  for datum in tqdm(full_data_selector, bar_format=short_progress_bar):
    if datum is None:
      continue
    rw_index = rw_rev_dict[datum.rewiring]
    dc_index = dc_rev_dict[datum.decay]
    bc_inst[rw_index][dc_index].append(datum)

  def eval_func(
      f_ext: Callable[[ScenarioStatistics], T1],
      f_sum: Callable[[List[T1]], T2],
  ):
    bc_inst_num: List[List[List[T1]]] = []
    for l_rw in bc_inst:
      bc_inst_num.append(
          [[xx for xx in (f_ext(x) for x in y) if xx is not None]
           for y in l_rw]
      )
    bc_inst_avg = [
        [f_sum(y) for y in x] for x in bc_inst_num
    ]
    return bc_inst_avg

  return eval_func


_heatmap_dict = {}


def create_heatmap_evaluator(
    session: Session,
    rs_name: str,
    retweet_rate: float,
):
  key = (id(session), rs_name, retweet_rate)
  if key in _heatmap_dict:
    return _heatmap_dict[key]
  val = create_heatmap_evaluator_raw(session, rs_name, retweet_rate)
  _heatmap_dict[key] = val
  return val


_session: Session | None = None


def get_session() -> Session:
  global _session
  assert _session is not None, "Session is not initialized"
  return _session


def heatmap_diff(
    f_ext: Callable[[ScenarioStatistics], float | None],
    f_sum: Callable[[List[float]], float],
    v_min=0.,
    v_max=1.,
):
  fig, axes = plt_figure(
      n_row=4, n_col=4, hw_ratio=1
  )
  axes_flattened: List[Axes] = []
  for a in axes:
    axes_flattened.extend(a)

  cmap_arr4, cmap_setter4 = get_colormap(
      axes_flattened, cmap='YlGnBu', fig=fig, anchor='W',
      vmin=v_min, vmax=v_max,
  )

  for i_rs, rs in enumerate(rs_keys):
    for i_tw, tw in enumerate(tw_keys):
      print('#', i_rs * 4 + i_tw + 1, rs, tw)
      axis = axes[i_rs][i_tw]
      if i_rs == 0 or i_tw == 0:
        title = f'{rs} / {tw}'
        axis.set_title(title, loc='left')

      eval_func = create_heatmap_evaluator(get_session(), rs, tw)
      heatmap = np.array(eval_func(
          f_ext, f_sum,
      ))
      axis.imshow(heatmap, **cmap_arr4)  # type: ignore

  fig.tight_layout()

  for axis in axes_flattened:
    axis.invert_yaxis()
    axis.set_xticks(np.arange(cfg.decay_rate_array.size))
    axis.set_xticklabels([' ' for _ in cfg.decay_rate_array])
    axis.set_yticks(np.arange(cfg.rewiring_rate_array.size))
    axis.set_yticklabels([' ' for _ in cfg.rewiring_rate_array])
    axis.grid(False)
  for axis in axes[3]:
    axis.set_xlabel('decay')
    axis.set_xticklabels(cfg.decay_rate_array, rotation=90)
  for axis, *_ in axes:
    axis.set_ylabel('rewiring')
    axis.set_yticklabels(cfg.rewiring_rate_array)

  cmap_setter4()

  return fig


def flatten_list(l: List[List[T1]]) -> List[T1]:
  return [item for sublist in l for item in sublist]


def curve_diff(
    f_ext: Callable[[ScenarioStatistics], Tuple[float, np.ndarray, np.ndarray] | None],
    x_label='x',
    y_label="y",
    f_0=0.,
    f_1=1.,
    eps=0.1,
    s_x=1.,
    s_y=1.,
):

  # build colormap

  c_a = (0, 0, 1, 0.1)    # 蓝色
  c_b = (1, 0, 0, 0.1)    # 红色

  cmap = LinearSegmentedColormap.from_list('my_cmap', [c_a, c_b])

  # s_y = 0.4

  fig, axes = plt_figure(
      n_row=4, n_col=4, hw_ratio=1
  )
  axes_flattened: List[Axes] = []
  for a in axes:
    axes_flattened.extend(a)

  xyc_groups = []
  for i_rs, rs in enumerate(rs_keys):
    for i_tw, tw in enumerate(tw_keys):
      print('#', i_rs * 4 + i_tw + 1, rs, tw)
      axis = axes[i_rs][i_tw]
      if i_rs == 0 or i_tw == 0:
        title = f'{rs} / {tw}'
        axis.set_title(title, loc='left')

      eval_func = create_heatmap_evaluator(get_session(), rs, tw)
      heatmap = eval_func(
          f_ext,
          lambda x: x,
      )
      for _h in heatmap:
        for __h in _h:
          for __h_el in __h:
            if __h_el is None:
              continue
            g, x, y = __h_el
            if g is None or x is None or y is None:
              continue
            c_g = cmap((g - f_0) / (f_1 - f_0))
            xyc_groups.append((axis, x, y, c_g))

  random.shuffle(xyc_groups)
  for axis, x, y, c_g in xyc_groups:
    axis.plot(x, y, color=c_g, linewidth=0.2)

  fig.tight_layout()

  for axis in axes_flattened:
    axis.set_xbound(-eps * s_x, s_x + eps * s_x)
    axis.set_ybound(-eps * s_y, s_y + eps * s_y)
    axis.grid(True)
  for axis in axes[3]:
    axis.set_xlabel(x_label)
  for axis, *_ in axes:
    axis.set_ylabel(y_label)

  return fig


def scatter_diff(
    # c, x, y
    f_ext: Callable[[ScenarioStatistics], Tuple[float, float, float] | None],
    x_label='x',
    y_label="y",
    f_0=0.,
    f_1=1.,
    eps=0.05,
    s_x=1.,
    s_y=1.,
    s_x_start=0.,
    s_y_start=0.,
):

  # build colormap

  c_a = (0, 0, 1, 0.1)    # 蓝色
  c_b = (1, 0, 0, 0.1)    # 红色

  cmap = LinearSegmentedColormap.from_list('my_cmap', [c_a, c_b])

  # s_y = 0.4

  fig, axes = plt_figure(
      n_row=4, n_col=4, hw_ratio=1
  )
  axes_flattened: List[Axes] = []
  for a in axes:
    axes_flattened.extend(a)

  for i_rs, rs in enumerate(rs_keys):
    for i_tw, tw in enumerate(tw_keys):
      print('#', i_rs * 4 + i_tw + 1, rs, tw)
      axis = axes[i_rs][i_tw]
      if i_rs == 0 or i_tw == 0:
        title = f'{rs} / {tw}'
        axis.set_title(title, loc='left')

      eval_func = create_heatmap_evaluator(get_session(), rs, tw)
      heatmap = eval_func(
          f_ext,
          lambda x: np.array(x).T,
      )
      heatmap_cat = np.concatenate(flatten_list(heatmap), axis=1, dtype=float)
      g, x, y = heatmap_cat
      c_g = cmap((g - f_0) / (f_1 - f_0))
      # xyc_groups.append((axis, x, y, c_g))
      axis.scatter(x, y, c=c_g, s=0.8)

  fig.tight_layout()

  s_x_int = s_x - s_x_start
  s_y_int = s_y - s_y_start

  for axis in axes_flattened:
    axis.set_xbound(s_x_start - eps * s_x_int, s_x + eps * s_x_int)
    axis.set_ybound(s_y_start - eps * s_y_int, s_y + eps * s_y_int)
    axis.grid(True)
  for axis in axes[3]:
    axis.set_xlabel(x_label)
  for axis, *_ in axes:
    axis.set_ylabel(y_label)

  return fig


def plot_diff(
    # c, y
    f_ext: Callable[[ScenarioStatistics], Tuple[float, float]],
    x_label='x',
    y_label="y",
    f_0=0.,
    f_1=1.,
    eps=0.05,
    s_x=1.,
    s_y=1.,
    s_x_start=0.,
    s_y_start=0.,
):

  # build colormap

  c_a = (0, 0, 1, 0.1)    # 蓝色
  c_b = (1, 0, 0, 0.1)    # 红色

  # s_y = 0.4

  fig, axes = plt_figure(
      n_row=4, n_col=4, hw_ratio=1
  )
  axes_flattened: List[Axes] = []
  for a in axes:
    axes_flattened.extend(a)

  for i_rs, rs in enumerate(rs_keys):
    for i_tw, tw in enumerate(tw_keys):
      print('#', i_rs * 4 + i_tw + 1, rs, tw)
      axis = axes[i_rs][i_tw]
      if i_rs == 0 or i_tw == 0:
        title = f'{rs} / {tw}'
        axis.set_title(title, loc='left')

      eval_func = create_heatmap_evaluator(get_session(), rs, tw)
      heatmap = eval_func(
          f_ext,
          lambda x: np.array(x).T,
      )
      heatmap_cat = np.concatenate(flatten_list(heatmap), axis=1)
      g, y = heatmap_cat
      g_mask_p1 = g >= 0.6
      g_mask_p2 = g < 0.6
      y1, y2 = y[g_mask_p1], y[g_mask_p2]
      axis.bar(1, np.mean(y1), yerr=np.std(y1),
               capsize=8, color=c_b, edgecolor='black')
      axis.bar(0, np.mean(y2), yerr=np.std(y2),
               capsize=8, color=c_a, edgecolor='black')

  fig.tight_layout()

  s_x_int = s_x - s_x_start
  s_y_int = s_y - s_y_start

  for axis in axes_flattened:
    # axis.set_xbound(s_x_start - eps * s_x_int, s_x + eps * s_x_int)
    axis.set_ybound(s_y_start - eps * s_y_int, s_y + eps * s_y_int)
    axis.grid(True)
  for axis in axes[3]:
    axis.set_xlabel(x_label)
  for axis, *_ in axes:
    axis.set_ylabel(y_label)

  return fig


if __name__ == '__main__':

  # TODO type check

  _session = create_db_session(stats_db_path, ScenarioStatistics.Base)

  stats_session = create_db_session(stats_db_path, ScenarioStatistics.Base)

  base_stats = True
  micro_stats = False

  if base_stats:

    fig_active_step = heatmap_diff(
        lambda x: x.active_step,
        lambda x: np.log10(np.mean(x)),
        v_min=1, v_max=5,
    )
    fig_active_step.show()

    fig_grad_index = heatmap_diff(
        lambda x: x.grad_index,
        lambda x: float(np.mean(x))
    )
    fig_grad_index.show()

    assert False

    fig_p_backdrop = heatmap_diff(
        lambda x: x.p_backdrop,
        lambda x: float(np.mean(x)),
        v_max=2,
    )
    fig_p_backdrop.show()

    fig_h_backdrop = heatmap_diff(
        lambda x: x.h_backdrop,
        lambda x: float(np.mean(x)),
        v_max=2,
    )
    fig_h_backdrop.show()

    assert False

    fig_c_grad_index = curve_diff(
        lambda x: (x.grad_index, x.p_index, x.h_index),
        x_label='polarization',
        y_label='homophily',
    )
    fig_c_grad_index.show()

    fig_c_homo_index = curve_diff(
        lambda x: (x.grad_index, x.x_indices / x.active_step, x.h_index),
        x_label='norm. time',
        y_label='homo. index',
    )
    fig_c_homo_index.show()

    fig_c_env_index = curve_diff(
        lambda x: (x.grad_index, x.x_indices / x.active_step, x.g_index),
        x_label='norm. time',
        y_label='env. index',
    )
    fig_c_env_index.show()

    fig_c_mean_vars = curve_diff(
        lambda x: (x.grad_index, x.x_mean_vars /
                   x.active_step, x.mean_vars_smpl),
        x_label='norm. time',
        y_label='mean var.',
        s_y=0.4,
    )
    fig_c_mean_vars.show()

  if micro_stats:

    def f_ext_env_index(x: ScenarioStatistics):
      f_init = piecewise_linear_integral_trapz(
          x.x_indices / x.active_step, x.g_index, 0, 1/3)
      f_final = piecewise_linear_integral_trapz(
          x.x_indices / x.active_step, x.g_index, 0, 1)
      if x.grad_index is None:
        return None
      return x.grad_index, f_init, f_final

    fig_s_env_index = scatter_diff(
        f_ext_env_index,
        s_x=0.3,
        s_y=0.9,
        s_y_start=0.2,
        x_label='int. (0~0.33)',
        y_label='int. (0~1)',
    )

    fig_s_env_index.show()

    def f_ext_event_count(x: ScenarioStatistics):
      if x.grad_index is None:
        return None
      return x.grad_index, np.log10(x.event_count), x.event_step_mean / x.active_step,

    fig_s_event_count = scatter_diff(
        f_ext_event_count,
        s_x_start=3.25,
        s_x=4.25,
        s_y=0.4,
        # s_y_start= 0.2,
        x_label='event count (log 10)',
        y_label='event time',
    )

    fig_s_event_count.show()

    fig_mean_vars = plot_diff(
        lambda x: (x.grad_index, piecewise_linear_integral_trapz(
            x.x_mean_vars / x.active_step, x.mean_vars_smpl, 0, 1)) if x.grad_index is not None else None,
        s_y=0.08,
        x_label='grad. index',
        y_label='mean var.',
    )
    fig_mean_vars.show()

    fig_triads = scatter_diff(
        lambda x: (x.grad_index, x.grad_index, np.log10(
            x.triads)) if x.grad_index is not None else None,
        s_y=5,
        s_y_start=3.5,
        x_label='grad. index',
        y_label='#closed triads (log 10)',

    )
