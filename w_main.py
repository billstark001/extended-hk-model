import os
import numpy as np
import time
from typing import Dict
from base import Scenario

from w_scenarios import all_scenarios, set_logger
import w_snapshots as ss

snapshot_interval = 4 * 60
max_snapshots = 3

from w_proc_utils import get_logger

ss.init()
logger = get_logger(__name__, os.path.join(ss.DEFAULT_ROOT_PATH, 'logfile.log'))
set_logger(logger.debug)

if __name__ == "__main__":

  for scenario_name, scenario_params in all_scenarios.items():

    scenario = Scenario(*scenario_params)

    def do_save():
      snapshot_name = ss.save(scenario.dump(), scenario_name)
      logger.info(
          f'Saved snapshot `{snapshot_name}` for scenario `{scenario_name}`. Model at step {scenario.steps}.')
      deleted = ss.delete_outdated(scenario_name, max_snapshots=max_snapshots)
      if deleted:
        logger.info(f'Deleted {deleted} outdated snapshots.')

    # recover or init model
    snapshot, snapshot_name = ss.load_latest(scenario_name)
    should_halt, max_edge, max_opinion = False, 0xffffff, 0xffffff
    if snapshot:
      scenario.load(*snapshot)
      
      should_halt, max_edge, max_opinion = scenario.check_halt_cond()
      if should_halt:
        logger.info(f'Simulation already finished for scenario `{scenario_name}`.')
        continue
      logger.info(
          f'Loaded snapshot `{snapshot_name}` for scenario `{scenario_name}`. Model at step {scenario.steps}.')
    else:
      scenario.init()
      snapshot_name = ss.save(scenario.dump(), scenario_name)
      logger.info(
          f'Initialized snapshot `{snapshot_name}` for scenario `{scenario_name}`.')

    last_timestamp = time.time()

    # simulate model
    errored = False
    while not should_halt:
      try:
        scenario.step_once()
        should_halt, max_edge, max_opinion = scenario.check_halt_cond()
        
        if scenario.steps % 100 == 0:
          logger.info(f'Model at step {scenario.steps}; max_edge={max_edge}, max_opinion={max_opinion}.')
        elif scenario.steps % 10 == 0:
          logger.debug(f'Model at step {scenario.steps}; max_edge={max_edge}, max_opinion={max_opinion}.')
        
        cur_timestamp = time.time()
        if cur_timestamp - last_timestamp > snapshot_interval:
          # save snapshot
          do_save()
          last_timestamp = cur_timestamp
        
        
      except KeyboardInterrupt:
        logger.info('KeyboardInterrupt detected. Model is halted.')
        do_save()
        errored = True
        break

      except Exception as e:
        logger.error(
            f"Simulation for `{scenario_name}` terminated unexpectedly at step {scenario.steps}.")
        logger.exception(e)
        errored = True
        break

    if not errored:
      logger.info(
          f'Simulation completed for scenario `{scenario_name}`. {scenario.steps} steps simulated in total.')
      do_save()
    else:
      break
