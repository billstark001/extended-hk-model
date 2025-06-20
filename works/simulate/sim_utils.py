
import logging
import os
import subprocess
import json
import time

from works.config import GO_SIMULATOR_PATH, GoMetadataDict

logger = logging.getLogger(__name__)


def simulate(
    sim_result_dir: str,
    params: GoMetadataDict,
    i: int | None = None,
    total_count: int | None = None,
):

  # check if finished
  _p = os.path.join(sim_result_dir, params['UniqueName'])
  if os.path.isdir(_p) and any(
      x.startswith('finished') for x in
      os.listdir(_p)
  ):
    if i is not None and (i + 1) % 50 == 0:
      print(f'Ignoring finished simulation: ({i + 1} / {total_count})')
    return False

  tstart = time.time()

  _cnt = (i + 1 if i is not None else 0, total_count or 0)

  logger.info(
      '(%d / %d) Scenario %s simulation started.',
      *_cnt, params['UniqueName']
  )

  params_json = json.dumps(params)

  params_proc = [GO_SIMULATOR_PATH, sim_result_dir, params_json]
  is_sim_halted = False
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

  log_params = [*_cnt, tdelta, params['UniqueName']]

  if returncode == 0:
    logger.info(
        '(%d / %d, %.2fs) %s: completed', *log_params
    )
  elif returncode == -15:
    logger.info(
        '(%d / %d, %.2fs) %s: halted by SIGTERM', *log_params
    )
  else:
    logger.error(
        '(%d / %d, %.2fs) %s: errored with code %d',
        *log_params, returncode
    )

  return is_sim_halted
