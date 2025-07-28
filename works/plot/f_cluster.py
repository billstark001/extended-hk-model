

from typing import Dict, List, Iterable, Callable, TypeVar, Tuple

import os

import numpy as np
from sqlalchemy import ColumnExpressionArgument
from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scipy.stats import gaussian_kde

from utils.plot import plt_figure
from utils.sqlalchemy import create_db_engine_and_session
import works.config as cfg
from works.plot.data_utils import piecewise_linear_integral_trapz
from works.stat.types import ScenarioStatistics


plot_path = cfg.SIMULATION_STAT_DIR
stats_db_path = os.path.join(plot_path, 'stats.db')

plt.rcParams.update({'font.size': 18})


def kde_min_bw_factory(min_bandwidth):
  def min_bw_factor(kde_obj):
    default_factor = kde_obj.scotts_factor()
    min_factor = min_bandwidth / np.std(kde_obj.dataset, ddof=1)
    return max(default_factor, min_factor)
  return min_bw_factor

if __name__ == '__main__':
  
  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base)
  
  
  stats_f1 = session.query(ScenarioStatistics).filter(
    ScenarioStatistics.grad_index.is_not(None),
    ScenarioStatistics.last_opinion_peak_count.is_not(None),
  )
  
  stats_grad, stats_peak, stats_retweet = np.array(
    [(x.grad_index, x.last_opinion_peak_count, x.retweet) for x in stats_f1],
    dtype=np.float64
  ).T
  
  stats_peak: np.ndarray = stats_peak.astype(int) # type: ignore
  
  
  resolution = 0.005
  margin = 0.3
  x = np.arange(-margin, 1 + margin + resolution, resolution)
  x_left = x < -resolution * 0.5
  x_right = x > 1 + resolution * 0.5
  x_mid = np.logical_not(np.logical_or(x_left, x_right))
  x_axis = x[x_mid]
  
  def fix_margin_stat(arr: np.ndarray):
    arr_left = arr[x_left][::-1]
    arr_right = arr[x_right][::-1]
    arr_mid = arr[x_mid]
    
    arr_mid[:arr_left.size] += arr_left
    arr_mid[-arr_right.size:] += arr_right
    
    return arr_mid
    
  # 1. div gross
  
  bw_method = kde_min_bw_factory(0.025)
  
  def _f(filter: np.ndarray):
    data_x = stats_grad[filter]
    if data_x.size == 0:
      return x_axis * 0
    return fix_margin_stat(
      gaussian_kde(
        data_x, 
        bw_method=bw_method
      )(x)
    )
  
  def draw_prob(
    filter_1: np.ndarray,
    filter_2: np.ndarray,
    filter_3: np.ndarray,
    total_size: int,
    ax: Axes,
  ):
    
    div_ratio_1 = np.sum(filter_1) / total_size
    div_ratio_2 = np.sum(filter_2) / total_size
    div_ratio_3 = np.sum(filter_3) / total_size
    
    
    div_group_1 = _f(filter_1) * div_ratio_1
    div_group_2 = _f(filter_2) * div_ratio_2
    div_group_3 = _f(filter_3) * div_ratio_3
    
    ax.fill_between(
      x_axis, x_axis * 0, div_group_1,
      label='#peaks = 1'
    )
    ax.fill_between(
      x_axis, div_group_1, div_group_1 + div_group_2,
      label='#peaks = 2'
    )
    ax.fill_between(
      x_axis, div_group_1 + div_group_2, div_group_1 + div_group_2 + div_group_3,
      label='#peaks >= 3'
    )
    
  
  
  fig, axes = plt_figure(
    n_col=3, n_row=1, total_width=24,
  )
  (ax1, ax2, ax3) = axes
  
  draw_prob(
    stats_peak == 1,
    stats_peak == 2,
    stats_peak > 2,
    stats_peak.size,
    ax1,
  )
  draw_prob(
    np.logical_and(stats_peak == 1, stats_retweet == 0),
    np.logical_and(stats_peak == 2, stats_retweet == 0),
    np.logical_and(stats_peak > 2, stats_retweet == 0),
    np.sum(stats_retweet == 0),
    ax2,
  )
  draw_prob(
    np.logical_and(stats_peak == 1, stats_retweet != 0),
    np.logical_and(stats_peak == 2, stats_retweet != 0),
    np.logical_and(stats_peak > 2, stats_retweet != 0),
    np.sum(stats_retweet != 0),
    ax3,
  )
  
  
  # ax1.plot(x_axis, div_group_1)
  # ax1.plot(x_axis, div_group_1 + div_group_2)
  # ax1.plot(x_axis, div_group_1 + div_group_2 + div_group_3)
  for ax in axes:
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('gradation index')
    
  ax1.set_ylabel('probability')
  
  ax1.set_title('(a) gross', loc='left')
  ax2.set_title('(b) no retweet', loc='left')
  ax3.set_title('(c) with retweet', loc='left')
  
  
  fig.show()

  session.close()
  engine.dispose()