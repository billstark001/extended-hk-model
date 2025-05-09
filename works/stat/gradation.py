
from typing import List, Any, Dict

import os
import json

import numpy as np
import matplotlib as mpl
from tqdm import tqdm

from result_interp.record import RawSimulationRecord
from utils.stat import compress_array_to_b64

import works.config as cfg
from works.stat.context import c


# parameters

scenario_base_path = cfg.SIMULATION_RESULT_DIR
plot_path = cfg.SIMULATION_PLOT_DIR

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

  itr = tqdm(cfg.all_scenarios_grad, bar_format=short_progress_bar)
  for scenario_metadata in itr:

    scenario_name = scenario_metadata['UniqueName']
    itr.set_postfix_str(scenario_name)

    if scenario_name in processed_data:
      pat_stats_set.append(processed_data[scenario_name])
      continue

    if unsaved:
      save_stats()

    # load scenario

    scenario_record = RawSimulationRecord(
        scenario_base_path,
        scenario_metadata,
    )
    scenario_record.load()
    if not scenario_record.is_finished:
      continue
    assert scenario_record.is_sanitized, 'non-sanitized scenario'

    c.set_state(
        scenario_record=scenario_record,
    )

    opinion_last_diff = c.opinion_last_diff
    event_step_mean = np.mean(c.event_step)

    bc_hom_last = c.bc_hom_last
    if np.isnan(bc_hom_last):
      bc_hom_last = None

    pat_stats = dict(
        name=scenario_name,
        step=c.total_steps,

        active_step=c.active_step,
        active_step_threshold=c.active_step_threshold,
        g_index_mean_active=c.g_index_mean_active,

        x_indices=c.x_indices,
        h_index=c.h_index, # hom index
        p_index=c.p_index, # pol index
        g_index=c.g_index, # env index

        grad_index=c.gradation_index_hp,
        event_count=c.event_step.size,
        event_step_mean=event_step_mean,
        triads=c.n_triads,

        bc_hom_last=bc_hom_last,

        x_bc_hom=c.x_bc_hom,
        bc_hom_smpl=c.bc_hom_smpl,

        x_mean_vars=c.x_mean_vars,
        mean_vars_smpl=c.mean_vars_smpl,

        opinion_diff=opinion_last_diff if np.isfinite(
            opinion_last_diff) else -1,
        
        x_opinion_diff_mean=c.x_opinion_diff_mean,
        opinion_diff_mean_smpl=c.opinion_diff_mean_smpl,

    )

    pat_stats_set.append(pat_stats)
    save_stats()

  if unsaved:
    save_stats()
