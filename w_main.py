import os
import numpy as np
import time
from typing import Dict
from base import Scenario

from w_scenarios import all_scenarios
import w_snapshots as ss

snapshot_interval = 4 * 60
max_snapshots = 3

from w_logger import logger

if __name__ == "__main__":

  for scenario_name, scenario_params in all_scenarios.items():

    model = Scenario(*scenario_params)

    def do_save():
      snapshot_name = ss.save(model.dump(), scenario_name)
      logger.info(
          f'Saved snapshot `{snapshot_name}` for scenario `{scenario_name}`. Model at step {model.steps}.')
      deleted = ss.delete_outdated(scenario_name, max_snapshots=max_snapshots)
      if deleted:
        logger.info(f'Deleted {deleted} outdated snapshots.')

    # recover or init model
    snapshot, snapshot_name = ss.load_latest(scenario_name)
    should_halt, max_edge, max_opinion = False, 0xffffff, 0xffffff
    if snapshot:
      model.load(*snapshot)
      
      should_halt, max_edge, max_opinion = model.check_halt_cond()
      if should_halt:
        logger.info(f'Simulation already finished for scenario `{scenario_name}`.')
        continue
      logger.info(
          f'Loaded snapshot `{snapshot_name}` for scenario `{scenario_name}`. Model at step {model.steps}.')
    else:
      model.init()
      snapshot_name = ss.save(model.dump(), scenario_name)
      logger.info(
          f'Initialized snapshot `{snapshot_name}` for scenario `{scenario_name}`.')

    last_timestamp = time.time()

    # simulate model
    errored = False
    while not should_halt:
      try:
        model.step_once()
        should_halt, max_edge, max_opinion = model.check_halt_cond()
        
        if model.steps % 100 == 0:
          logger.info(f'Model at step {model.steps}; max_edge={max_edge}, max_opinion={max_opinion}.')
        elif model.steps % 10 == 0:
          logger.debug(f'Model at step {model.steps}; max_edge={max_edge}, max_opinion={max_opinion}.')
        
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
            f"Simulation for `{scenario_name}` terminated unexpectedly at step {model.steps}.")
        logger.exception(e)
        errored = True
        break

    if not errored:
      logger.info(
          f'Simulation completed for scenario `{scenario_name}`. {model.steps} steps simulated in total.')
      do_save()
    else:
      break
