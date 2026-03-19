import os

from utils.file import init_logger
from works.config import SMP_BINARY_PATH, all_scenarios_eps, get_workspace_dir
from smp_bindings.simulation import run_simulations

SIMULATION_RESULT_DIR = get_workspace_dir()

os.makedirs(SIMULATION_RESULT_DIR, exist_ok=True)
init_logger(None, os.path.join(
    SIMULATION_RESULT_DIR, 'logfile.log'))

if __name__ == '__main__':

  print(f'Result Directory: {SIMULATION_RESULT_DIR}')

  run_simulations(SMP_BINARY_PATH, SIMULATION_RESULT_DIR, all_scenarios_eps)
