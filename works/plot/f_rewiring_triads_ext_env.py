from typing import Dict, List, Iterable, Callable, TypeVar, Tuple

import os

import numpy as np
from sqlalchemy import ColumnExpressionArgument
from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from utils.plot import plt_figure
from utils.sqlalchemy import create_db_engine_and_session
import works.config as cfg
from works.plot.data_utils import piecewise_linear_integral_trapz
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

k_filters = [
    (
        ScenarioStatistics.recsys_type == 'StructureM9',
        ScenarioStatistics.retweet == 0,
    ),
    (
        ScenarioStatistics.recsys_type == 'StructureM9',
        ScenarioStatistics.retweet != 0,
    ),
    (
        ScenarioStatistics.recsys_type == 'OpinionM9',
        ScenarioStatistics.retweet == 0,
    ),
    (
        ScenarioStatistics.recsys_type == 'OpinionM9',
        ScenarioStatistics.retweet != 0,
    ),
]

k_labels = [
    'St / !R',
    'St / R',
    'Op / !R',
    'Op / R',
]

m_filters = [
    (
        ScenarioStatistics.grad_index.is_not(None),
        ScenarioStatistics.grad_index < 0.6,
    ),
    (
        ScenarioStatistics.grad_index.is_not(None),
        ScenarioStatistics.grad_index > 0.6,
    ),
]

m_labels = [
    'polarized',
    'homogenized',
]

m_colors = [
  'tab:blue',
  'tab:red'
]


def evaluate_bar_on_filters(
    session: Session,
    filters: List[ColumnExpressionArgument[bool]],
    f_ext: Callable[[ScenarioStatistics], float],
):
  full_data: Iterable[ScenarioStatistics] = session.query(
      ScenarioStatistics).filter(*filters)

  data_arr: List[float] = []
  for datum in full_data:
    data_arr.append(f_ext(datum))

  return np.mean(data_arr), np.std(data_arr)


def plot_bar(
    ax: Axes,
    f: Callable[[ScenarioStatistics], float]
):

  k = len(k_filters)
  m = len(m_filters)

  all_data: List[List[Tuple[float, float]]] = []

  for k_filter in k_filters:
    row_data: List[Tuple[float, float]] = []
    for m_filter in m_filters:
      filters = [*k_filter, *m_filter]
      mean, std = evaluate_bar_on_filters(session, filters, f)
      row_data.append((mean, std))  # type: ignore
    all_data.append(row_data)

  all_data_arr = np.array(all_data, dtype=float)  # (k, m, [mean, std])
  means = all_data_arr[:, :, 0]
  stds = all_data_arr[:, :, 1]

  width = 0.6 / m  # width of the bars
  x = np.arange(k)

  for i in range(m):
    ax.bar(
        x + i * width,
        means[:, i],
        width,
        yerr=stds[:, i],
        label=m_labels[i],
        capsize=4,
        edgecolor='black', 
        color=m_colors[i],
        alpha=0.5,
    )

  ax.set_xticks(x + width*(m-1)/2)
  ax.set_xticklabels(k_labels)

  return ax


if __name__ == '__main__':

  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base)
  
  # rewiring events
  
  fig, (ax1, ax2) = plt_figure(n_row=1, n_col=2, total_width=16)

  def f_ev_count(x: ScenarioStatistics) -> float:
    assert x.event_count is not None
    return np.log10(x.event_count)

  def f_ev_step(x: ScenarioStatistics) -> float:
    assert x.event_step_mean is not None
    assert x.active_step is not None
    return x.event_step_mean / max(x.active_step, 1)
  
  ext_env_perp_vec = np.array([-1/3, 1])
  ext_env_perp_vec /= np.linalg.norm(ext_env_perp_vec) 
  # this one already normalized
  
  def f_ext_env_index(x: ScenarioStatistics) -> float:
    assert x.x_indices is not None
    assert x.active_step is not None
    assert x.g_index is not None
    f_init = piecewise_linear_integral_trapz(x.x_indices / x.active_step, x.g_index, 0, 1/3)
    f_final =  piecewise_linear_integral_trapz(x.x_indices / x.active_step, x.g_index, 0, 1)
    
    f_vec = np.array([f_init, f_final])
    
    f_proj = np.dot(f_vec, ext_env_perp_vec) 
    
    return f_proj

  plot_bar(ax1, f=f_ev_count)
  plot_bar(ax2, f=f_ev_step)
  
  ax1.set_title('(a) log10(event count)', loc='left')
  ax2.set_title('(b) avg. event time', loc='left')
  
  ax1.grid(True, linestyle='--', alpha=0.5)
  ax2.grid(True, linestyle='--', alpha=0.5)
  
  ax1.legend()
  ax1.set_ylim(2.5, 4.5)
  
  ax2.legend()
  ax2.set_ylim(0, 0.4)
  
  fig.show()
  
  # triads
  
  fig2, ax3 = plt_figure(n_row=1, n_col=1, total_width=8)
  
  def f_triads(x: ScenarioStatistics) -> float:
    assert x.triads is not None
    return np.log10(x.triads)
  
  plot_bar(ax3, f=f_triads)
  ax3.set_title('log10(#closed triads)', loc='left')
  ax3.legend()
  
  ax3.grid(True, linestyle='--', alpha=0.5)
  ax3.set_ylim(3.5, 5)
  
  fig2.show()
  
  # ext env index
  
  fig3, ax4 = plt_figure(n_row=1, n_col=1, total_width=8)
  
  plot_bar(ax4, f_ext_env_index)
  
  ax4.set_title('TODO what title should we use?')
  ax4.legend()
  ax4.grid(True, linestyle='--', alpha=0.5)
  
  ax4.set_ylim(0.3, 0.7)
  
  fig3.show()

  session.close()
  engine.dispose()
