from typing import Dict, List, TypedDict

import os

import dotenv
import numpy as np

# build scenarios

decay_rate_array = rewiring_rate_array = \
    np.array([0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1])
retweet_rate_array = np.array([0, 0.1, 0.25, 0.5])

tolerance_array = np.array([
    0.05, 0.1, 0.15, 0.2,
    0.25, 0.3, 0.35, 0.4,
    0.45, 0.5, 0.55, 0.6,
    0.7, 0.8, 0.9, 1
])

n_sims = 10
n_sims_eps = 100
n_sims_rep = 100

rs_names = {
    'st': ('StructureM9', 0),
    'op0': ('OpinionM9', 0),
    'op2': ('OpinionM9', 2),
    'op6': ('OpinionM9', 6),
}

# all parameters


class GoMetadataDict(TypedDict):
  UniqueName: str
  Tolerance: float
  Decay: float
  RewiringRate: float
  RetweetRate: float
  RecsysFactoryType: str
  RecsysCount: int
  TweetRetainCount: int
  MaxSimulationStep: int


def create_go_metadata_dict(
    name: str,
    tolerance=0.45,
    decay=0.9,
    rewiring=0.01,
    retweet=0.05,
    recsys_type="Random",
    recsys_count=10,
    tweet_retain_count=3,
    max_sim_step=20000,
) -> GoMetadataDict:
  return {
      "UniqueName": name,
      "Tolerance": tolerance,
      "Decay": decay,
      "RewiringRate": rewiring,
      "RetweetRate": retweet,
      "RecsysFactoryType": recsys_type,
      "RecsysCount": recsys_count,
      "TweetRetainCount": tweet_retain_count,
      "MaxSimulationStep": max_sim_step,
  }


all_scenarios_grad: List[GoMetadataDict] = []

for i_sim in range(n_sims):
  for i_rw, rw in enumerate(rewiring_rate_array):
    for i_dc, dc in enumerate(decay_rate_array):
      for i_rt, rt in enumerate(retweet_rate_array):
        for k_rs, (rs, t_retain) in rs_names.items():
          x = create_go_metadata_dict(
              f's_grad_sim{i_sim}_rw{i_rw}_dc{i_dc}_rt{i_rt}_{k_rs}',
              rewiring=rw,
              decay=dc,
              retweet=rt,
              recsys_type=rs,
              tweet_retain_count=t_retain,
          )
          all_scenarios_grad.append(x)


all_scenarios_eps: List[GoMetadataDict] = []

for i_sim in range(n_sims_eps):
  for i_to, to in enumerate(tolerance_array):
    x = create_go_metadata_dict(
        f's_eps_sim{i_sim}_to{i_to}',
        rewiring=0.05,
        decay=0.05,
        retweet=0.1,
        recsys_type='Random',
        tweet_retain_count=3,
        tolerance=to,
        max_sim_step=20000,
    )
    all_scenarios_eps.append(x)


all_scenarios_rep: List[GoMetadataDict] = []

for i_sim in range(n_sims_rep):
  for i_rs, rs in [('rn', 'Random'), ('st', 'StructureM9')]:
    x = create_go_metadata_dict(
        f's_rep_sim{i_sim}_{i_rs}',
        rewiring=0.05,
        decay=0.05,
        retweet=0.1,
        recsys_type=rs,
        tweet_retain_count=3,
        tolerance=0.45,
        max_sim_step=20000,
    )
    all_scenarios_rep.append(x)


# assign paths
dotenv.load_dotenv()


def normalize_path(p: str):
  if p == '' or p == None:
    raise ValueError("Empty path")
  p = os.path.expandvars(p)
  p = os.path.expanduser(p)
  p = os.path.abspath(p)
  p = os.path.normpath(p)
  # if os.name == 'nt':
  #   p = p.replace('\\', '/')
  return p


def normalize_int(p: str | None, default: int = 0):
  if not p:
    return default
  try:
    return int(p.strip())
  except ValueError:
    pass  # do nothing
  return default


GO_SIMULATOR_PATH = normalize_path('./ehk-model/main')
SIMULATION_RESULT_DIR = normalize_path(os.environ["SIMULATION_RESULT_DIR"])

SIMULATION_PLOT_DIR = normalize_path(os.environ['SIMULATION_PLOT_DIR'])

SIMULATION_INSTANCE_NAME = os.environ['SIMULATION_INSTANCE_NAME']
assert SIMULATION_INSTANCE_NAME is not None and SIMULATION_INSTANCE_NAME != '', 'SIMULATION_INSTANCE_NAME not defined'

STAT_THREAD_COUNT = max(normalize_int(
    os.environ.get('STAT_THREAD_COUNT', None), 6), 1)
