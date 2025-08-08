

from typing import Dict, List, Iterable, Callable, Any, Tuple

import os

import numpy as np
from sqlalchemy import ColumnExpressionArgument

from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scipy.stats import gaussian_kde

from utils.plot import plt_figure, plt_save_and_close
from utils.stat import adaptive_moving_stats
from utils.sqlalchemy import create_db_engine_and_session
import works.config as cfg
from works.plot.data_utils import piecewise_linear_integral_trapz
from works.stat.types import ScenarioStatistics


plot_path = cfg.SIMULATION_STAT_DIR
stats_db_path = os.path.join(plot_path, 'stats.db')

plt.rcParams.update({
  'font.size': 18,
  # 'font.family': 'serif',
  # 'font.serif': 'Computer Modern Roman',
})


def kde_min_bw_factory(min_bandwidth):
  def min_bw_factor(kde_obj):
    default_factor = kde_obj.scotts_factor()
    min_factor = min_bandwidth / np.std(kde_obj.dataset, ddof=1)
    return max(default_factor, min_factor)
  return min_bw_factor


def plot_bar(
    ax: Axes,
    all_data: List[List[Tuple[float, float]]] | np.ndarray,
    k_labels: List[str],
    m_labels: List[str],
    m_colors: List[Any],
):
  
  all_data_arr = np.array(all_data, dtype=float)  # (k, m, [mean, std])
  k, m, _ = all_data_arr.shape
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
  
  
  stats_raw = session.query(ScenarioStatistics).filter(
    ScenarioStatistics.grad_index.is_not(None),
    ScenarioStatistics.opinion_diff_seg_mean.is_not(None),
    ScenarioStatistics.opinion_diff_seg_std.is_not(None),
  )
  
  def _filter(f: Callable[[ScenarioStatistics], bool]):
  
    stats_filtered = [x for x in stats_raw if f(x)]
    
    raw_mean, raw_std = np.array([(x.opinion_diff_seg_mean, x.opinion_diff_seg_std) for x in stats_filtered]).transpose(1, 0, -1)
    opinion_diff_mean = np.mean(raw_mean, axis=0)
    
    std_term_1 = raw_std ** 2
    std_term_2 = (raw_mean - opinion_diff_mean) ** 2
    
    var_term = np.mean((std_term_1 + std_term_2), axis=0)
    opinion_diff_std = var_term ** 0.5
    
    return opinion_diff_mean, opinion_diff_std
    
  od_p2_r_mean, od_p2_r_std = _filter(lambda x: x.grad_index < 0.6 and x.retweet != 0) # type: ignore
  od_p2_nr_mean, od_p2_nr_std = _filter(lambda x: x.grad_index < 0.6 and x.retweet == 0) # type: ignore
  od_p1_r_mean, od_p1_r_std = _filter(lambda x: x.grad_index >= 0.6 and x.retweet != 0) # type: ignore
  od_p1_nr_mean, od_p1_nr_std = _filter(lambda x: x.grad_index >= 0.6 and x.retweet == 0) # type: ignore
  
  # contrast
  
  x1, y1, y_std_1 = np.array([(x.grad_index, x.opinion_diff_seg_mean[1], x.opinion_diff_seg_std[1]) for x in stats_raw if x.retweet == 0]).T # type: ignore
  # plt.scatter(x1, y1, s=0.01)

  x1_f, y1_f, _ = adaptive_moving_stats(x1, y1, 0.2, min=0, max=1)
  _, y1_std_f, _ = adaptive_moving_stats(x1, y_std_1, 0.2, min=0, max=1)

  x2, y2, y_std_2 = np.array([(x.grad_index, x.opinion_diff_seg_mean[1], x.opinion_diff_seg_std[1]) for x in stats_raw if x.retweet != 0]).T # type: ignore

  x2_f, y2_f, _ = adaptive_moving_stats(x2, y2, 0.2)
  _, y2_std_f, _ = adaptive_moving_stats(x2, y_std_2, 0.2, min=0, max=1)
  
  
  # plot
  
  fig, axes = plt_figure(n_row=1, n_col=3, total_width=24)
  ax1, ax2, ax3 = axes
  
  x_labels = [
    '0~.2',
    '.2~.4',
    '.4~.6',
    '.6~.8',
    '.8~1'
  ]
  
  plot_bar(
    ax1,
    np.array([[od_p2_nr_mean, od_p2_nr_std], [od_p2_r_mean, od_p2_r_std]]).transpose(2, 0, 1),
    x_labels,
    ['no retweet', 'retweet'],
    ['tab:blue', 'tab:cyan']
  )
  
  plot_bar(
    ax2,
    np.array([[od_p1_nr_mean, od_p1_nr_std], [od_p1_r_mean, od_p1_r_std]]).transpose(2, 0, 1),
    x_labels,
    ['no retweet', 'retweet'],
    ['tab:red', 'tab:orange']
  )
  
  
  ax3.plot(x1_f, y1_f, label='no retweet')
  ax3.fill_between(x1_f, y1_f - y1_std_f, y1_f + y1_std_f, alpha=0.1)
  ax3.plot(x2_f, y2_f, label='retweet')
  ax3.fill_between(x2_f, y2_f - y2_std_f, y2_f + y2_std_f, alpha=0.1)
  
  for ax in axes:
    ax.grid(True)
    ax.legend()
  
  for ax in (ax1, ax2):
    ax.set_xlabel('normalized time')
  ax3.set_xlabel('gradation index')
  
  ax1.set_ylabel('mean opinion difference')
  
  ax1.set_title('(a) polarization-dom.', loc='left')
  ax2.set_title('(b) homophily-dom.', loc='left')
  ax3.set_title('(c) t_a = 0.2~0.4', loc='left')
    
  ax1.set_ylim(0, 0.6)
  ax2.set_ylim(0, 0.5)
  
  
  plt_save_and_close(fig, 'fig/f_retweet_opinion_diff')

  session.close()
  engine.dispose()