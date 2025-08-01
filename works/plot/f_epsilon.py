from typing import Iterable, TypeVar

import os

import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy.orm import Session

from utils.plot import plt_figure, plt_save_and_close
from utils.sqlalchemy import create_db_session
import works.config as cfg
from works.stat.types import ScenarioStatistics


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

# parameters

plot_path = cfg.SIMULATION_STAT_DIR
stats_db_path = os.path.join(plot_path, 'stats.eps.db')

rs_keys = list(cfg.rs_names)
tw_keys = cfg.retweet_rate_array.tolist()


plt.rcParams.update({'font.size': 18})

# utilities

# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


def get_random_stats(
    session: Session,
    decay: float | None = None,
    rewiring: float | None = None,
    retweet: float | None = None,
):
  full_data_selector: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.name.startswith('s_eps'),
      ScenarioStatistics.recsys_type == 'Random',
      *(x for x in (
          (ScenarioStatistics.decay == decay) if decay is not None else None,
          (ScenarioStatistics.rewiring == rewiring) if rewiring is not None else None,
          (ScenarioStatistics.retweet == retweet) if retweet is not None else None,
      ) if x is not None)
  )
  return list(full_data_selector)


if __name__ == '__main__':

  stats_session = create_db_session(stats_db_path, ScenarioStatistics.Base)

  stats = get_random_stats(stats_session)

  all_epsilon_vals = set(x.tolerance for x in stats)
  vals = {x: [] for x in sorted(list(all_epsilon_vals))}
  for x in stats:
    vals[x.tolerance].append(
        (x.grad_index, x.last_community_count, x.last_opinion_peak_count))

  plt_x = np.array(list(vals.keys()))
  plt_ig_avg, plt_cm_avg, plt_cp_avg, \
      plt_ig_std, plt_cm_std, plt_cp_std = np.array(
          [(np.mean(v, axis=0), np.std(v, axis=0)) for v in vals.values()]
      ).reshape(-1, 6).T

  fig, (ax1, ax2, ax3) = plt_figure(n_row=1, n_col=3, total_width=24)

  ax1.errorbar(plt_x, plt_ig_avg, yerr=plt_ig_std, zorder=1)
  ax2.errorbar(plt_x, plt_cm_avg, yerr=plt_cm_std, zorder=1)
  ax3.errorbar(plt_x, plt_cp_avg, yerr=plt_cp_std, zorder=1)

  ax1.scatter(plt_x, plt_ig_avg, marker='x', c='black', zorder=2)
  ax2.scatter(plt_x, plt_cm_avg, marker='x', c='black', zorder=2)
  ax3.scatter(plt_x, plt_cp_avg, marker='x', c='black', zorder=2)

  ax1.grid(True)
  ax2.grid(True)
  ax3.grid(True)

  ax1.set_title('(a) Gradation Index')
  ax2.set_title('(b) #Community (Leiden)')
  ax3.set_title('(c) #Opinion Peaks')

  plt_save_and_close(fig, 'fig/f_eps_select')
