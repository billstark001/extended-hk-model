from typing import Dict, Iterable, List, Tuple

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from utils.plot import plt_figure, plt_save_and_close, setup_paper_params
from utils.sqlalchemy import create_db_engine_and_session
import works.config as cfg
from works.stat.types import ScenarioStatistics


plot_path = cfg.SIMULATION_STAT_DIR
stats_db_path = os.path.join(plot_path, "stats.db")

retweet_groups: List[float] = cfg.retweet_rate_array.tolist()
rs_groups: List[Tuple[str, int]] = [
    (rs, int(retain)) for rs, retain in cfg.rs_names.values()
]


def _safe_curve(
    p_index: np.ndarray | None,
    h_index: np.ndarray | None,
) -> np.ndarray | None:
  if p_index is None or h_index is None:
    return None
  if p_index.size < 2 or h_index.size < 2:
    return None

  n = min(p_index.size, h_index.size)
  p = p_index[:n]
  h = h_index[:n]

  mask = np.isfinite(p) & np.isfinite(h)
  if np.count_nonzero(mask) < 2:
    return None

  p_valid = np.minimum(np.maximum(p[mask], 0.0), 1.0)
  h_valid = np.minimum(np.maximum(h[mask], 0.0), 1.0)
  points = np.column_stack((p_valid, h_valid))
  if points.shape[0] < 2:
    return None
  return points


def _group_data(
    full_data: Iterable[ScenarioStatistics],
) -> Dict[Tuple[float, str, int], List[Tuple[np.ndarray, float]]]:
  grouped: Dict[Tuple[float, str, int], List[Tuple[np.ndarray, float]]] = {}
  for rt in retweet_groups:
    for rs_type, retain_count in rs_groups:
      grouped[(rt, rs_type, retain_count)] = []

  for datum in full_data:
    if datum.grad_index is None:
      continue
    if datum.p_index is None or datum.h_index is None:
      continue

    retain_count = int(round(float(datum.tweet_retain_count)))
    key = (float(datum.retweet), str(datum.recsys_type), retain_count)
    if key not in grouped:
      continue

    points = _safe_curve(datum.p_index, datum.h_index)
    if points is None:
      continue

    grad = min(max(float(datum.grad_index), 0.0), 1.0)
    grouped[key].append((points, grad))

  return grouped


def plot_p_h_index_trajectories() -> Figure:
  setup_paper_params()

  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base
  )

  full_data: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.name.startswith("s_grad"),
      ScenarioStatistics.p_index.is_not(None),
      ScenarioStatistics.h_index.is_not(None),
      ScenarioStatistics.grad_index.is_not(None),
  )

  grouped = _group_data(full_data)

  fig, axes = plt_figure(
      n_row=4,
      n_col=4,
      hw_ratio=1,
      constrained_layout=False,
  )

  axes_arr = np.array(axes)
  cmap = cm.get_cmap("coolwarm")
  norm = Normalize(vmin=0, vmax=1)

  for i_rt, rt in enumerate(retweet_groups):
    for i_rs, (rs_type, retain_count) in enumerate(rs_groups):
      ax = axes_arr[i_rt, i_rs]
      curves = grouped[(rt, rs_type, retain_count)]

      for points, grad in curves:
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=cmap(norm(grad)),
            linewidth=0.35,
            alpha=0.8,
        )

      ax.set_xlim(0, 1)
      ax.set_ylim(0, 1)
      ax.grid(True, linestyle="--", linewidth=0.1, alpha=0.35)

      title_rs = "St" if rs_type == "StructureM9" else "Op"
      ax.set_title(
          f"p={rt:g}, {title_rs}/k={retain_count} (n={len(curves)})",
          loc="left",
          fontsize=9,
      )

      if i_rt == 3:
        ax.set_xlabel(r"$I_p$")
      else:
        ax.set_xticklabels([])

      if i_rs == 0:
        ax.set_ylabel(r"$I_h$")
      else:
        ax.set_yticklabels([])

  sm = cm.ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  fig.colorbar(sm, ax=axes_arr.ravel().tolist(), fraction=0.02, pad=0.01, label=r"$I_w$")

  fig.tight_layout()

  session.close()
  engine.dispose()
  return fig


if __name__ == "__main__":
  plt_save_and_close(
      plot_p_h_index_trajectories(),
      "fig/f_p_h_index_trajectory",
  )
