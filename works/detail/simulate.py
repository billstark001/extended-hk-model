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


from utils.stat import get_logger


BASE_PATH = './run3'
os.makedirs(BASE_PATH, exist_ok=True)

logger = get_logger(__name__, os.path.join(BASE_PATH, 'logfile.log'))

# build scenarios


def stat_collectors_f():
  ret = {
    # 'triads': stats.TriadsCountCollector(),
    'layout': stats.NetworkLayoutCollector(return_dict=True),
    's-index': stats.SegregationIndexCollector(),
    'distance': stats.DistanceCollectorDiscrete(
        use_js_divergence=True,
        hist_interval=0.08,
    ),
  }
  return ret

def save_sim_result(S: Scenario, name: str):
  # dump_data = S.dump()
  dump_data = S.generate_record_data()
  with open(os.path.join(BASE_PATH, name + '_record.pkl'), 'wb') as f:
    pickle.dump(dump_data, f)


def check_sim_result(name: str):
  return os.path.exists(os.path.join(BASE_PATH, name + '.pkl'))


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
    opinion_change_error=1e-5,
    model_stat_collectors=stat_collectors_f(),
    agent_stat_keys=['cur_opinion', 'nr_agents', 'op_sum_agents', 'follow_event'],
)


decay_rewiring_segs = 7
decay_rewiring_groups = (10 ** np.vstack([
  np.linspace(-1, -2, decay_rewiring_segs),
  np.linspace(-2, -1, decay_rewiring_segs)
])).T

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


n_sims = 20

# all parameters

params_arr: List[Tuple[str, float, float, float]] = []

for i_sim in range(n_sims):
  for i_dr, (d, r) in enumerate(decay_rewiring_groups):
    for i_g, g in enumerate(n_gens):
      g_name = n_gen_names[i_g]
      x = (
          f'scenario_i{len(params_arr)}_dr{i_dr}_{g_name}_sim{i_sim}',
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

    sim_p_standard.model_stat_collectors = stat_collectors_f()
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
