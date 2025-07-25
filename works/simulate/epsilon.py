import os

from utils.stat import init_logger
from works.config import all_scenarios_eps, get_workspace_dir
from works.simulate.sim_utils import simulate

SIMULATION_RESULT_DIR = get_workspace_dir()

os.makedirs(SIMULATION_RESULT_DIR, exist_ok=True)
init_logger(None, os.path.join(
    SIMULATION_RESULT_DIR, 'logfile.log'))

if __name__ == '__main__':

  print(f'Result Directory: {SIMULATION_RESULT_DIR}')

  total_count = len(all_scenarios_eps)
  is_sim_halted = False
  for i, params in enumerate(all_scenarios_eps):

    is_sim_halted = simulate(SIMULATION_RESULT_DIR, params, i, total_count)
    if is_sim_halted:
      break
