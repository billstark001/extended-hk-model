from typing import Dict, List

import os

import dotenv
import numpy as np

# build scenarios

decay_rate_array = rewiring_rate_array = \
    np.array([0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1])
n_sims = 50

n_gen_names = {
  'op': 'OpinionM9',
  'st': 'StructureM9',
}

# all parameters

def create_go_metadata_dict(
  name: str,
  tolerance = 0.45,
  decay = 0.9,
  rewiring = 0.01,
  retweet = 0.05,
  recsys_type = "Random",
  recsys_count = 3,
):
  return {
    "UniqueName": name,
    "Tolerance": tolerance,
    "Decay": decay,
    "RewiringRate": rewiring,
    "RetweetRate": retweet,
    "RecsysFactoryType": recsys_type,
    "RecsysCount": recsys_count,
  }

all_scenarios_grad: List[Dict] = []

for i_sim in range(n_sims):
  for i, r in enumerate(rewiring_rate_array):
    for j, d in enumerate(decay_rate_array):
      for k, g in n_gen_names.items():
        x = create_go_metadata_dict(
          f's_grad_r{i}_d{j}_{k}_sim{i_sim}',
          rewiring=r,
          decay=d,
          recsys_type=g,
        )
        all_scenarios_grad.append(x)


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

GO_SIMULATOR_PATH = normalize_path('./ehk-model/main')
SIMULATION_RESULT_DIR = normalize_path(os.environ["SIMULATION_RESULT_DIR"])
SIMULATION_TEMP_FILE = os.path.join(
  SIMULATION_RESULT_DIR,
  os.environ['SIMULATION_TEMP_FILE_NAME']
)

SIMULATION_PLOT_DIR = normalize_path(os.environ['SIMULATION_PLOT_DIR'])
