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
from recsys import Random, Opinion, Structure, Mixed
import stats

from w_logger import logger

BASE_PATH = './run2'
os.makedirs(BASE_PATH, exist_ok=True)

# build scenarios


def stat_collectors_f(): return {
    'layout': stats.NetworkLayoutCollector(use_last=True),
    'triads': stats.TriadsCountCollector(),
    'cluster': stats.ClusteringCollector(),
    's-index': stats.SegregationIndexCollector(),
    'in-degree': stats.InDegreeCollector(),
    'distance': stats.DistanceCollectorDiscrete(
        use_js_divergence=True,
        hist_interval=0.08,
    ),
}


def save_sim_result(S: Scenario, name: str):
  dump_data = S.dump()
  with open(os.path.join(BASE_PATH, name + '.pkl'), 'wb') as f:
    pickle.dump(dump_data, f)


network_provider = RandomNetworkProvider(
    agent_count=400,
    agent_follow=15,
)
sim_p_standard = SimulationParams(
    max_total_step=10000,
    stat_interval=20,
    opinion_change_error=1e-4,
    stat_collectors=stat_collectors_f()
)


# rewiring_rate_array = 10 ** np.arange(-2, 0 + 1/5, 1/4)
# decay_rate_array = 10 ** np.arange(-3, 0, 1/3)

# rewiring_rate_array = np.array([0.01, 0.03, 0.05, 0.1, 0.3, 0.5])
# decay_rate_array = np.array([0.005, 0.01, 0.05, 0.1, 0.5, 1])

decay_rate_array = rewiring_rate_array = \
    np.array([0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1])
n_sims = 5

n_gens = [
    # lambda m: Random(m),
    lambda m: Mixed(
        m,
        Random(m, 10),
        Opinion(m),
        0.1),
    lambda m: Mixed(
        m,
        Random(m, 10),
        Structure(m, sigma=0.2, matrix_init=True),
        0.1),
]

n_gen_names = ['op', 'st']

# all parameters

params_arr: List[Tuple[str, float, float,
                       Callable[[HKModel], HKModelRecommendationSystem]]] = []

for i_sim in range(n_sims):
  for i, r in enumerate(rewiring_rate_array):
    for j, d in enumerate(decay_rate_array):
      for k, g in enumerate(n_gens):
        x = (
            f'scenario_i{len(params_arr)}_r{i}_d{j}_{n_gen_names[k]}_sim{i_sim}',
            r,
            d,
            g,
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
  for scenario_name, r, d, g in params_arr[::-1]:
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
