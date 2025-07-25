
import os

import numpy as np

from result_interp.record import RawSimulationRecord
import works.config as cfg
from works.stat.context import c
from works.stat.types import ScenarioStatistics

from works.stat.tasks import generate_stats


def get_statistics(
    scenario_metadata: 'cfg.GoMetadataDict',
    scenario_base_path: str,
    origin: str,
    exist_stats: ScenarioStatistics | None,
    active_threshold=0.98,
    min_inactive_value=0.75,
):

  scenario_name = scenario_metadata['UniqueName']

  scenario_record = RawSimulationRecord(
      scenario_base_path,
      scenario_metadata,
  )
  scenario_record.load()
  if not scenario_record.is_finished:
    return None

  assert scenario_record.is_sanitized, 'non-sanitized scenario'

  c.set_state(
      scenario_record=scenario_record,
      active_threshold=active_threshold,
      min_inactive_value=min_inactive_value,
  )
  event_step_mean = np.mean(c.event_step)

  pat_stats = ScenarioStatistics(
      id=exist_stats.id if exist_stats else None,
      name=scenario_name,
      origin=origin,

      tolerance=scenario_metadata['Tolerance'],
      decay=scenario_metadata['Decay'],
      rewiring=scenario_metadata['RewiringRate'],
      retweet=scenario_metadata['RetweetRate'],
      recsys_type=scenario_metadata['RecsysFactoryType'],
      tweet_retain_count=scenario_metadata['TweetRetainCount'],

      step=c.total_steps,
      active_step=c.active_step,
      active_step_threshold=c.active_step_threshold,
      g_index_mean_active=c.g_index_mean_active,

      x_indices=c.x_indices,
      h_index=c.h_index,  # hom index
      p_index=c.p_index,  # pol index
      g_index=c.g_index,  # env index

      grad_index=c.gradation_index_hp,
      event_count=c.event_step.size,
      event_step_mean=event_step_mean,
      triads=c.n_triads,

      x_mean_vars=c.x_mean_vars,
      mean_vars_smpl=c.mean_vars_smpl,

      last_community_count=c.last_community_count,
      last_community_sizes=c.last_community_sizes,

      last_opinion_peak_count=c.last_opinion_peak_count,
  )

  return pat_stats


# parameters

scenario_base_path = cfg.get_workspace_dir()
plot_path = cfg.SIMULATION_STAT_DIR

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

stats_db_path = os.path.join(plot_path, 'stats.db')


# utilities

# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


c.set_state(
    active_threshold=0.98,
    min_inactive_value=0.75
)


def stats_exist(name: str, origin: str) -> bool:
  try:
    ScenarioStatistics.get(
        ScenarioStatistics.name == name,
        ScenarioStatistics.origin == origin,
    )
    return True
  except ScenarioStatistics.DoesNotExist:
    return False


if __name__ == '__main__':

  generate_stats(
      get_statistics,
      scenario_base_path,
      stats_db_path,
      cfg.get_instance_name(),
      cfg.all_scenarios_eps,
  )
