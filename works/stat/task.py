# type: ignore

from typing import Type, TYPE_CHECKING

import numpy as np

from result_interp.record import RawSimulationRecord

from works.stat.context import c

import peewee

from utils.peewee import NumpyArrayField, nullable


@nullable(exclude=[
    'name', 'origin',
    'tolerance', 'decay', 'rewiring', 'retweet',
    'recsys_type', 'tweet_retain_count',
])
class ScenarioStatistics(peewee.Model):
  DoesNotExist: Type[peewee.DoesNotExist]

  class Meta:
    database: peewee.Database
  _meta: Type[Meta]

  # model metadata
  name: str = peewee.CharField(max_length=255, index=True)
  origin: str = peewee.CharField(max_length=255, index=True)

  tolerance: float = peewee.DoubleField(index=True)  # epsilon
  decay: float = peewee.DoubleField(index=True)  # alpha
  rewiring: float = peewee.DoubleField(index=True)  # q
  retweet: float = peewee.DoubleField(index=True)  # p
  recsys_type: str = peewee.CharField(max_length=255, index=True)
  tweet_retain_count: float = peewee.DoubleField(index=True)

  # simulation result
  step: int = peewee.IntegerField()
  active_step: int = peewee.IntegerField()
  active_step_threshold: float = peewee.DoubleField()

  g_index_mean_active: float = peewee.DoubleField()

  x_indices: np.ndarray = NumpyArrayField(dtype=np.int32)
  h_index: np.ndarray = NumpyArrayField(dtype=np.float64)  # hom index
  p_index: np.ndarray = NumpyArrayField(dtype=np.float64)  # pol index
  g_index: np.ndarray = NumpyArrayField(dtype=np.float64)  # env index

  grad_index: float = peewee.DoubleField()
  event_count: int = peewee.IntegerField()
  event_step_mean: float = peewee.DoubleField()
  triads: int = peewee.IntegerField()

  x_bc_hom: np.ndarray = NumpyArrayField(dtype=np.int32)
  bc_hom_smpl: np.ndarray = NumpyArrayField(dtype=np.float64)

  x_mean_vars: np.ndarray = NumpyArrayField(dtype=np.int32)
  mean_vars_smpl: np.ndarray = NumpyArrayField(dtype=np.float64)

  opinion_diff: float = peewee.DoubleField()

  x_opinion_diff_mean: np.ndarray = NumpyArrayField(dtype=np.int32)
  opinion_diff_mean_smpl: np.ndarray = NumpyArrayField(dtype=np.float64)


if TYPE_CHECKING:
  from works.config import GoMetadataDict


def stats_from_dict(
    scenario_metadata: 'GoMetadataDict',
    stats_dict: dict,
    origin: str,
):

  _d: dict = {}
  for k, v in stats_dict.items():
    if k in [
        'name', 'origin',
        'tolerance', 'decay', 'rewiring', 'retweet',
        'recsys_type', 'tweet_retain_count',
    ]:
      continue
    if hasattr(ScenarioStatistics, k):
      _d[k] = v

  scenario_name = scenario_metadata['UniqueName']
  pat_stats = ScenarioStatistics(
      name=scenario_name,
      origin=origin,

      tolerance=scenario_metadata['Tolerance'],
      decay=scenario_metadata['Decay'],
      rewiring=scenario_metadata['RewiringRate'],
      retweet=scenario_metadata['RetweetRate'],
      recsys_type=scenario_metadata['RecsysFactoryType'],
      tweet_retain_count=scenario_metadata['TweetRetainCount'],

      **_d,
  )
  return pat_stats


def get_statistics(
    scenario_metadata: 'GoMetadataDict',
    scenario_base_path: str,
    origin: str,
    active_threshold=0.98,
    min_inactive_value=0.75,
):

  scenario_name = scenario_metadata['UniqueName']

  # if stats_exist(scenario_name, origin):
  #   return None

  # load scenario

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

  opinion_last_diff = c.opinion_last_diff
  event_step_mean = np.mean(c.event_step)

  bc_hom_last = c.bc_hom_last
  if np.isnan(bc_hom_last):
    bc_hom_last = None

  pat_stats = ScenarioStatistics(
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

      x_bc_hom=c.x_bc_hom,
      bc_hom_smpl=c.bc_hom_smpl,

      x_mean_vars=c.x_mean_vars,
      mean_vars_smpl=c.mean_vars_smpl,

      opinion_diff=opinion_last_diff if np.isfinite(
          opinion_last_diff) else -1,

      x_opinion_diff_mean=c.x_opinion_diff_mean,
      opinion_diff_mean_smpl=c.opinion_diff_mean_smpl,

  )

  return pat_stats
