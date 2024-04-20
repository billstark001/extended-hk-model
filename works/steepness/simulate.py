from typing import Callable, Mapping, cast, Optional, List, Tuple
from numpy.typing import NDArray

import os

import pickle
from matplotlib.axes import Axes
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

from base import HKModelParams, Scenario, SimulationParams
from base.model import HKModel, HKModelRecommendationSystem
from env import RandomNetworkProvider, ScaleFreeNetworkProvider
from recsys import Random, OpinionRandom
import stats


from utils.stat import get_logger


BASE_PATH = './run3'
os.makedirs(BASE_PATH, exist_ok=True)

logger = get_logger(__name__, os.path.join(BASE_PATH, 'logfile.log'))

# build scenarios


def stat_collectors_f(): return {
    'triads': stats.TriadsCountCollector(),
    'cluster': stats.ClusteringCollector(),
    's-index': stats.SegregationIndexCollector(),
    'distance': stats.DistanceCollectorDiscrete(
        use_js_divergence=True,
        hist_interval=0.08,
    ),
}


def save_sim_result(S: Scenario, name: str):
  dump_data = S.dump()
  with open(os.path.join(BASE_PATH, name + '.pkl'), 'wb') as f:
    pickle.dump(dump_data, f)


def check_sim_result(name: str):
  return os.path.exists(os.path.join(BASE_PATH, name + '.pkl'))


network_provider = RandomNetworkProvider(
    agent_count=400,
    agent_follow=15,
)
sim_p_standard = SimulationParams(
    max_total_step=15000,
    stat_interval=20,
    opinion_change_error=1e-4,
    stat_collectors=stat_collectors_f()
)

steepness_array = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
rewiring_rate_array = decay_rate_array = np.array([0.05, 0.1, 0.3])


def get_recsys_factory(steepess: float):
  if steepess <= 0:
    return lambda m: Random(m)
  return lambda m: OpinionRandom(
      m,
      tolerance=0.45,
      steepness=steepess,
  )


n_sims = 20

# all parameters

params_arr: List[Tuple[str, float, float,
                       Callable[[HKModel], HKModelRecommendationSystem]]] = []

for i_sim in range(n_sims):
  for i_s, s in enumerate(steepness_array):
    for i_d, d in enumerate(decay_rate_array):
      for i_r, r in enumerate(rewiring_rate_array):
        x = (
            f'scenario_i{len(params_arr)}_s{i_s}_d{i_d}_r{i_r}_sim{i_sim}',
            s,
            d,
            r,
            get_recsys_factory(s)
        )
        params_arr.append(x)


def gen_params(r: float, d: float, g: float):
  return HKModelParams(
      tolerance=0.45,
      decay=d,
      rewiring_rate=r,
      recsys_count=10,
      recsys_factory=g
  )


if __name__ == '__main__':
  for scenario_name, s, d, r, g in params_arr:
    if check_sim_result(scenario_name):
      continue

    params = gen_params(r, d, g)

    sim_p_standard.stat_collectors = stat_collectors_f()
    scenario = Scenario(network_provider, params, sim_p_standard)
    scenario.init()

    logger.info('Scenario %s simulation started.', scenario_name)

    try:

      scenario.step()
      save_sim_result(scenario, scenario_name)

      logger.info('Saved scenario %s. Model at step %d.',
                  scenario_name, scenario.steps)

    except Exception as e:
      logger.error(
          'Error occurred when simulating scenario %s.', scenario_name)
      logger.exception(e)
