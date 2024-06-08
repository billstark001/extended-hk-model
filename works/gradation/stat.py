
from typing import List, Any, Dict

import os

import json
import pickle
import importlib

import numpy as np
from tqdm import tqdm

import matplotlib as mpl

from utils.stat import compress_array_to_b64
import works.gradation.simulate as p

from works.stat import c


# parameters

scenario_base_path = './run2'
plot_path = './fig2'

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

pat_stats_path = os.path.join(plot_path, 'pattern_stats.json')


_b = compress_array_to_b64
dummy_comp_arr = _b(np.zeros((0,), dtype=int))


# utilities

mpl.rcParams['font.size'] = 18


# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


c.set_state(
    active_threshold=0.98,
    min_inactive_value=0.75
)


if __name__ == '__main__':

  pat_stats_set: List[Dict[str, Any]] = []
  processed_data = {}

  if os.path.exists(pat_stats_path):
    with open(pat_stats_path, 'r', encoding='utf8') as fp:
      lst = json.load(fp)
      for d in lst:
        processed_data[d['name']] = d

  unsaved = False

  def save_stats():
    global unsaved
    with open(pat_stats_path, 'w', encoding='utf8') as f:
      json.dump(pat_stats_set, f, indent=2, ensure_ascii=False)
    unsaved = False

  for scenario_name, r, d, g in tqdm(p.params_arr, bar_format=short_progress_bar):

    if scenario_name in processed_data:
      pat_stats_set.append(processed_data[scenario_name])
      continue

    if unsaved:
      save_stats()

    # load scenario

    scenario_path = os.path.join(
        scenario_base_path, scenario_name + '_record.pkl')
    if not os.path.exists(scenario_path):
      continue

    with open(scenario_path, 'rb') as f:
      S_metadata, S_stats, S_agent_stats = pickle.load(f)

    c.set_state(
        model_metadata=S_metadata,
        model_stats=S_stats,
        agent_stats=S_agent_stats,
    )

    opinion_last_diff = c.opinion_last_diff
    pat_stats = dict(
        name=scenario_name,
        step=c.total_steps,

        active_step=c.active_step,
        active_step_threshold=c.active_step_threshold,
        g_index_mean_active=c.g_index_mean_active,

        p_last=c.p_index[-1],
        h_last=c.h_index[-1],
        s_last=c.s_index[-1],

        pat_area_hp=c.gradation_index_hp,

        triads2=c.n_triads,

        in_degree=c.in_degree,
        opinion_diff=opinion_last_diff if np.isfinite(
            opinion_last_diff) else -1,

        smpl_pearson_rel=c.smpl_pearson_rel,
        smpl_rec_dis_network=c.smpl_rec_dis_network,
        smpl_rec_concordant_n=c.smpl_rec_concordant_n,
    )

    pat_stats_set.append(pat_stats)
    save_stats()

    print(scenario_name)

  if unsaved:
    save_stats()
