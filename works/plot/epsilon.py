from typing import Dict, List, Iterable, Callable, TypeVar, Tuple

import os
import copy
import random

import numpy as np
import peewee
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.plot import plt_figure, get_colormap
from utils.peewee import sync_peewee_table
import works.config as cfg
from works.stat.types import ScenarioStatistics
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import interpolate, integrate


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


def get_random_stats(
  decay: float | None = None,
  rewiring: float | None = None,
  retweet: float | None = None,
):
  full_data_selector: Iterable[ScenarioStatistics] = ScenarioStatistics.select().where(
    ScenarioStatistics.name.startswith('s_eps'),
    ScenarioStatistics.recsys_type == 'Random',
    *(x for x in (
      (ScenarioStatistics.decay == decay) if decay is not None else None,
      (ScenarioStatistics.rewiring == rewiring) if rewiring is not None else None,
      (ScenarioStatistics.retweet == retweet) if retweet is not None else None,
    ) if x is not None)
  )
  return list(full_data_selector)

def piecewise_linear_integral_trapz(x: np.ndarray, y: np.ndarray, a: float, b: float):
  # 保证a < b
  if a > b:
      a, b = b, a

  # 合并端点
  x_new = np.sort(np.concatenate([x, [a, b]]))
  y_new = np.interp(x_new, x, y, left=0, right=1)

  # 只保留在[a, b]区间的点
  mask = (x_new >= a) & (x_new <= b)
  x_new = x_new[mask]
  y_new = y_new[mask]

  # 积分
  return np.trapz(y_new, x_new)

if __name__ == '__main__':

  stats_db = peewee.SqliteDatabase(stats_db_path)
  stats_db.connect()

  ScenarioStatistics._meta.database = stats_db
  stats_db.create_tables([ScenarioStatistics])
  
  stats = get_random_stats()
  
  all_epsilon_vals = set(x.tolerance for x in stats)
  vals = { x: [] for x in sorted(list(all_epsilon_vals)) }
  for x in stats:
    vals[x.tolerance].append((x.grad_index, x.last_community_count))
    
  plt_x = np.array(list(vals.keys()))
  plt_ig_avg, plt_cm_avg, plt_ig_std, plt_cm_std = np.array(
    [(np.mean(v, axis=0), np.std(v, axis=0)) for v in vals.values()]
  ).reshape(-1, 4).T
  
  
  fig, (ax1, ax2) = plt_figure(n_row=1, n_col=2)
  
  ax1.errorbar(plt_x, plt_ig_avg, yerr=plt_ig_std)
  ax2.errorbar(plt_x, plt_cm_avg, yerr=plt_cm_std)

  ax1.grid(True)
  ax2.grid(True)

  fig.show()  
  