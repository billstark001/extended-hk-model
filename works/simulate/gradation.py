import os
import subprocess
import json
import time

from utils.stat import get_logger
from works.config import GO_SIMULATOR_PATH, all_scenarios_grad, SIMULATION_RESULT_DIR, SIMULATION_TEMP_FILE

os.makedirs(SIMULATION_RESULT_DIR, exist_ok=True)
logger = get_logger(__name__, os.path.join(SIMULATION_RESULT_DIR, 'logfile.log'))

if __name__ == '__main__':
  
  total_count = len(all_scenarios_grad)
  is_sim_halted = False
  for i, params in enumerate(all_scenarios_grad):
    tstart = time.time()
    
    logger.info(
      '(%d / %d) Scenario %s simulation started.', 
      i + 1, total_count, params['UniqueName']
    )
    
    os.makedirs(SIMULATION_RESULT_DIR, exist_ok=True)
    
    with open(SIMULATION_TEMP_FILE, 'w') as f:
      json.dump(params, f)
      
    params_proc = [GO_SIMULATOR_PATH, SIMULATION_RESULT_DIR, SIMULATION_TEMP_FILE]
    with subprocess.Popen(
      params_proc,
      text=True     
    ) as proc:
      try:
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

    tdelta = time.time() - tstart
    
    log_params = [i + 1, total_count, tdelta, params['UniqueName']]

    if returncode == 0:
      logger.info('(%d / %d, %.2fs) %s: completed', *log_params)
    elif returncode == -15:
      logger.info('(%d / %d, %.2fs) %s: halted by SIGTERM', *log_params)
    else:
      logger.error('(%d / %d, %.2fs) %s: errored with code %d', *log_params, returncode)
      
    if is_sim_halted:
      break
      