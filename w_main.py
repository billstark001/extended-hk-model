import os
import pickle
import time
from typing import Dict
from base import Scenario
import traceback

from w_scenarios import all_scenarios
import w_snapshots as ss

snapshot_interval = 2 * 60
max_snapshots = 5

from w_logger import logger

if __name__ == "__main__":

  for scenario_name, scenario_params in all_scenarios.items():

    model = Scenario(*scenario_params)

    def do_save():
      snapshot_name = ss.save(model.dump(), scenario_name)
      logger.info(
          f'Saved snapshot `{snapshot_name}` for scenario `{scenario_name}`. Model at step {model.steps}.')
      deleted = ss.delete_outdated(scenario_name)
      if deleted:
        logger.info(f'Deleted {deleted} outdated snapshots.')

    # recover or init model
    snapshot, snapshot_name = ss.load_latest(scenario_name)
    if snapshot:
      model.load(*snapshot)
      if model.should_halt():
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
    while not model.should_halt():
      try:
        model.step_once()
        
        if model.steps % 100 == 0:
          logger.info(f'Model at step {model.steps}.')
        elif model.steps % 10 == 0:
          logger.debug(f'Model at step {model.steps}.')
        
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
