from typing import Dict, List

import os
import subprocess
import json
import numpy as np

from utils.stat import get_logger

BASE_PATH = './run2'
os.makedirs(BASE_PATH, exist_ok=True)

logger = get_logger(__name__, os.path.join(BASE_PATH, 'logfile.log'))

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

params_arr: List[Dict] = []

for i_sim in range(n_sims):
  for i, r in enumerate(rewiring_rate_array):
    for j, d in enumerate(decay_rate_array):
      for k, g in n_gen_names.items():
        x = create_go_metadata_dict(
          f'scenario_i{len(params_arr)}_r{i}_d{j}_{k}_sim{i_sim}',
          rewiring=r,
          decay=d,
          recsys_type=g,
        )
        params_arr.append(x)

_p = lambda p: os.path.normpath(os.path.expanduser(p))

GO_SIMULATOR_PATH = _p('./ehk-model/main')
SIMULATION_RESULT_DIR = _p('./run/')
SIMULATION_TEMP_FILE = _p('./run/temp_metadata.json')

if __name__ == '__main__':
  total_count = len(params_arr)
  is_sim_halted = False
  for i, params in enumerate(params_arr):
    
    logger.info(
      '(%d / %d) Scenario %s simulation started.', 
      i + 1, total_count, params['UniqueName']
    )
    
    os.makedirs(SIMULATION_RESULT_DIR, exist_ok=True)
    
    with open(SIMULATION_TEMP_FILE, 'w') as f:
      json.dump(params, f)
      
    params_proc = [GO_SIMULATOR_PATH, SIMULATION_RESULT_DIR, SIMULATION_TEMP_FILE]
    print(' '.join(params_proc))
    with subprocess.Popen(
      params_proc,
      # stdout=subprocess.PIPE,
      # stderr=subprocess.PIPE,
      text=True     
    ) as proc:
      try:
        # stdout, stderr = proc.communicate()
        proc.wait()
        returncode = proc.returncode
      except KeyboardInterrupt:
        proc.terminate()
        try:
          proc.wait(timeout=60)
        except Exception:
          proc.kill()
        is_sim_halted = True
        returncode = -15

    if returncode == 0:
      logger.info('Simulation of scenario %s complete', params['UniqueName'])
    elif returncode == -15:
      logger.info('Simulation of scenario %s halted by SIGTERM', params['UniqueName'])
    else:
      logger.error('Simulation of scenario %s errored with code %d', params['UniqueName'], returncode)
      
    if is_sim_halted:
      break
      