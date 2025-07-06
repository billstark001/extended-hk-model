# type: ignore

from typing import Type, TYPE_CHECKING

import numpy as np
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

  last_community_count: int = peewee.IntegerField()
  last_community_sizes: str = peewee.TextField()

  last_opinion_peak_count: int = peewee.IntegerField()
  
  p_backdrop: float = peewee.DoubleField()
  h_backdrop: float = peewee.DoubleField()
  g_backdrop: float = peewee.DoubleField()


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
