from typing import Dict, List, Iterable, Callable, TypeVar, Tuple

import os

import numpy as np
from sqlalchemy import ColumnExpressionArgument
from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from utils.plot import plt_figure, plt_save_and_close, setup_paper_params
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


setup_paper_params()

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

k_filters_1 = [
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

k_labels_1 = [
    'St, !R',
    'St, R',
    'Op, !R',
    'Op, R',
]

k_filters_2 = [
  (
    ScenarioStatistics.recsys_type == 'OpinionM9',
    ScenarioStatistics.tweet_retain_count == 0,
        ScenarioStatistics.retweet == 0,
  ),
  (
    ScenarioStatistics.recsys_type == 'OpinionM9',
    ScenarioStatistics.tweet_retain_count == 2,
        ScenarioStatistics.retweet == 0,
  ),
  (
    ScenarioStatistics.recsys_type == 'OpinionM9',
    ScenarioStatistics.tweet_retain_count == 6,
        ScenarioStatistics.retweet == 0,
  ),
  (
    ScenarioStatistics.recsys_type == 'OpinionM9',
    ScenarioStatistics.tweet_retain_count == 0,
        ScenarioStatistics.retweet != 0,
  ),
  (
    ScenarioStatistics.recsys_type == 'OpinionM9',
    ScenarioStatistics.tweet_retain_count == 2,
        ScenarioStatistics.retweet != 0,
  ),
  (
    ScenarioStatistics.recsys_type == 'OpinionM9',
    ScenarioStatistics.tweet_retain_count == 6,
        ScenarioStatistics.retweet != 0,
  ),
]

k_labels_2 = [
  'k=0, !R',
  'k=2',
  'k=6',
  'k=0, R',
  'k=2',
  'k=6',
]



m_filters = [
    (
        ScenarioStatistics.grad_index.is_not(None),
        ScenarioStatistics.grad_index < 0.6,
    ),
    (
        ScenarioStatistics.grad_index.is_not(None),
        ScenarioStatistics.grad_index >= 0.6,
    ),
]

m_labels = [
    'd-pol.',
    'c-pol.',
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
    f: Callable[[ScenarioStatistics], float],
    compare_op = False,
    plot_title: str = "",
) -> List[str]:
  
  ret: List[str] = []
  
  k_filters = k_filters_2 if compare_op else k_filters_1
  k_labels = k_labels_2 if compare_op else k_labels_1

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

  # Print data for each bar
  if plot_title:
    ret.append(f"\n=== {plot_title} ===")
  for k_idx in range(k):
    for m_idx in range(m):
      ret.append(f"{k_labels[k_idx]} - {m_labels[m_idx]}: {means[k_idx, m_idx]:.4f} ± {stds[k_idx, m_idx]:.4f}")

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

  return ret


if __name__ == '__main__':

  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base)
  
  # Create figure with 2 rows: first row with 2 subplots, second row with 3 subplots
  fig, axes = plt_figure(n_row=2, n_col=3)
  ax1, ax2 = axes[0][0], axes[0][1]
  ax3, ax4, ax5 = axes[1][0], axes[1][1], axes[1][2]
  
  # Hide the unused subplot in the first row
  axes[0][2].set_visible(False)

  def f_ev_count(x: ScenarioStatistics) -> float:
    assert x.event_count is not None
    return x.event_count

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

  def f_triads(x: ScenarioStatistics) -> float:
    assert x.triads is not None
    return x.triads

  # Plot 1: Event count (1_1)
  data_1 = plot_bar(ax1, f=f_ev_count, plot_title="Event Count")
  ax1.set_title('(a) #rewiring events', loc='left')
  ax1.grid(True, linestyle='--', alpha=0.5)
  ax1.legend()
  ax1.set_yscale('log')
  ax1.set_ylim(100, 50000)
  
  # Plot 2: Event step (1_2)
  data_2 = plot_bar(ax2, f=f_ev_step, plot_title="Event Step")
  ax2.set_title('(b) norm. occurrence time', loc='left')
  ax2.grid(True, linestyle='--', alpha=0.5)
  ax2.legend()
  ax2.set_ylim(0, 0.4)
  
  # Plot 3: Front-loading environment index (3_1)
  data_3 = plot_bar(ax3, f_ext_env_index, plot_title="Front-loading Environment Index")
  ax3.set_title(r"(c) front-loading env. index $I_e'$", loc='left')
  ax3.grid(True, linestyle='--', alpha=0.5)
  ax3.legend()
  ax3.set_ylim(0.3, 0.7)
  
  # Plot 4: Triads (2_1)
  data_4 = plot_bar(ax4, f=f_triads, plot_title="Triads")
  ax4.set_title('(d) #triads', loc='left')
  ax4.grid(True, linestyle='--', alpha=0.5)
  ax4.legend()
  ax4.set_yscale('log')
  ax4.set_ylim(1000, 200000)
  
  # Plot 5: Triads for Opinion (2_2)
  data_5 = plot_bar(ax5, f=f_triads, compare_op=True, plot_title="Triads (Opinion)")
  ax5.set_title('(e) #triads, Op.', loc='left')
  ax5.grid(True, linestyle='--', alpha=0.5)
  ax5.legend()
  ax5.set_yscale('log')
  ax5.set_ylim(1000, 200000)
  
  # Save the combined figure
  fig.tight_layout()
  plt_save_and_close(fig, 'fig/f_grad_index_interpret_raw')
  
  with open('fig/f_grad_index_interpret_raw_data.txt', 'w') as f:
    for line in data_1 + data_2 + data_3 + data_4 + data_5:
      f.write(line + '\n')

  session.close()
  engine.dispose()
