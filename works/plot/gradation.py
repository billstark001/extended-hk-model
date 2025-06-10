from typing import Dict, List, Iterable, Callable, TypeVar

import os
import copy

import numpy as np
import peewee
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.plot import plt_figure, get_colormap
from utils.peewee import sync_peewee_table
import works.config as cfg
from works.stat.task import ScenarioStatistics
from matplotlib.colors import LinearSegmentedColormap, Normalize


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

# parameters

plot_path = cfg.SIMULATION_PLOT_DIR
stats_db_path = os.path.join(plot_path, 'stats.db')

rs_keys = list(cfg.rs_names)
tw_keys = cfg.retweet_rate_array.tolist()


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
  rs_name: str,
  retweet_rate: float,
):
  rs_key, rs_retain = cfg.rs_names[rs_name]
  full_data_selector: Iterable[ScenarioStatistics] = ScenarioStatistics.select().where(
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
        [[f_ext(x) for x in y] for y in l_rw]
      )
    bc_inst_avg = [
      [f_sum(y) for y in x] for x in bc_inst_num
    ]
    return bc_inst_avg
    
  return eval_func
  


def heatmap_diff():
  fig, axes = plt_figure(
    n_row=4, n_col=4, hw_ratio=1
  )
  axes_flattened: List[Axes] = []
  for a in axes:
    axes_flattened.extend(a)
    
  cmap_arr4, cmap_setter4 = get_colormap(
      axes_flattened, cmap='YlGnBu', fig=fig, anchor='W',
      # vmin=0.7, vmax=1,
      vmin = 1, vmax = 5,
  )
  
  for i_rs, rs in enumerate(rs_keys):
    for i_tw, tw in enumerate(tw_keys):
      print('#', i_rs * 4 + i_tw + 1, rs, tw)
      axis = axes[i_rs][i_tw]
      if i_rs == 0 or i_tw == 0:
        title = f'{rs} / {tw}'
        axis.set_title(title, loc='left')
      
      eval_func = create_heatmap_evaluator(rs, tw)
      heatmap = np.array(eval_func(
        lambda x: x.active_step,
        lambda x: np.log10(np.mean(x)),
      ))
      axis.imshow(heatmap, **cmap_arr4) # type: ignore
        

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


def curve_diff():
  
  # build colormap
    
  c_a = (0, 0, 1, 0.1)    # 蓝色
  c_b = (1, 0, 0, 0.1)    # 红色

  cmap = LinearSegmentedColormap.from_list('mycmap', [c_a, c_b])
  
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
      
      eval_func = create_heatmap_evaluator(rs, tw)
      heatmap = eval_func(
        # lambda x: (x.grad_index, x.p_index, x.h_index),
        lambda x: (x.grad_index, x.x_indices / x.active_step, x.h_index),
        # lambda x: (x.grad_index, x.x_mean_vars / x.active_step, x.mean_vars_smpl),
        lambda x: x,
      )
      for _h in heatmap:
        for __h in _h:
          for g, x, y in __h:
            c_g = cmap((g - 0.7) / (1 - 0.7))
            axis.plot(x, y, color=c_g, linewidth=0.5)
        

  fig.tight_layout()

  for axis in axes_flattened:
    eps = 0.1
    axis.set_xbound(-eps, 1 + eps)
    axis.set_ybound(-eps, 1 + eps)
    axis.grid(True)
  for axis in axes[3]:
    axis.set_xlabel('polarization')
  for axis, *_ in axes:
    axis.set_ylabel('homophily')
  # for axis in axes[3]:
  #   axis.set_xlabel('norm. time')
  # for axis, *_ in axes:
  #   axis.set_ylabel('env. index')
  
  return fig


if __name__ == '__main__':

  stats_db = peewee.SqliteDatabase(stats_db_path)
  stats_db.connect()

  ScenarioStatistics._meta.database = stats_db
  stats_db.create_tables([ScenarioStatistics])

  # fig = heatmap_diff()
  fig = curve_diff()
  fig.show()