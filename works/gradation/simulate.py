from typing import Callable, Mapping, cast, Optional, List, Tuple
from numpy.typing import NDArray

import os
# import sys
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)

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


from utils.stat import get_logger


BASE_PATH = './run2'
os.makedirs(BASE_PATH, exist_ok=True)

logger = get_logger(__name__, os.path.join(BASE_PATH, 'logfile.log'))

# build scenarios


def stat_collectors_f(layout=False):
  ret = {
    # 'triads': stats.TriadsCountCollector(),
    # 'cluster': stats.ClusteringCollector(),
    's-index': stats.SegregationIndexCollector(),
    'in-degree': stats.InDegreeCollector(),
    'distance': stats.DistanceCollectorDiscrete(
        use_js_divergence=True,
        hist_interval=0.08,
    ),
  }
  if layout:
    ret['layout'] = stats.NetworkLayoutCollector(return_dict=True)
  return ret


def save_sim_result(S: Scenario, name: str):
  dump_data = S.generate_record_data()
  with open(os.path.join(BASE_PATH, name + '_record.pkl'), 'wb') as f:
    pickle.dump(dump_data, f)

def check_sim_result(name: str):
  return os.path.exists(os.path.join(BASE_PATH, name + '_record.pkl'))

network_provider = RandomNetworkProvider(
    agent_count=400,
    agent_follow=15,
)
sim_p_standard = SimulationParams(
    max_total_step=15000,
    model_stat_interval={
        50: 5,
        150: 10,
        300: 15,
        114514: 20,
    },
    opinion_change_error=1e-4,
    model_stat_collectors=stat_collectors_f(layout=True),
    agent_stat_keys=['cur_opinion', 'nr_agents', 'op_sum_agents', 'follow_event'],
)


# rewiring_rate_array = 10 ** np.arange(-2, 0 + 1/5, 1/4)
# decay_rate_array = 10 ** np.arange(-3, 0, 1/3)

# rewiring_rate_array = np.array([0.01, 0.03, 0.05, 0.1, 0.3, 0.5])
# decay_rate_array = np.array([0.005, 0.01, 0.05, 0.1, 0.5, 1])

decay_rate_array = rewiring_rate_array = \
    np.array([0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1])
n_sims = 20

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
        Structure(m, noise_std=0.2, matrix_init=True),
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
  for scenario_name, r, d, g in params_arr:
    if check_sim_result(scenario_name):
      continue
    
    params = gen_params(r, d, g)

    sim_p_standard.model_stat_collectors = stat_collectors_f(layout=True)
    scenario = Scenario(network_provider, params, sim_p_standard)
    scenario.init()

    logger.info('Scenario %s simulation started.', scenario_name)

    try:

      scenario.iter()
      save_sim_result(scenario, scenario_name)

      logger.info('Saved scenario %s. Model at step %d.',
                  scenario_name, scenario.steps)

    except Exception as e:
      logger.error(
          'Error occurred when simulating scenario %s.', scenario_name)
      logger.exception(e)
