from typing import Dict

import os
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

import peewee
from tqdm import tqdm


from utils.peewee import sync_peewee_table
import works.config as cfg
from works.stat.context import c
from works.stat.task import ScenarioStatistics, get_statistics


# parameters

scenario_base_path = cfg.SIMULATION_RESULT_DIR
plot_path = cfg.SIMULATION_PLOT_DIR

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

stats_db_path = os.path.join(plot_path, 'stats.db')


# utilities

# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


c.set_state(
    active_threshold=0.98,
    min_inactive_value=0.75
)


def stats_exist(name: str, origin: str) -> bool:
  try:
    ScenarioStatistics.get(
        ScenarioStatistics.name == name,
        ScenarioStatistics.origin == origin,
    )
    return True
  except ScenarioStatistics.DoesNotExist:
    return False


if __name__ == '__main__':

  stats_db = peewee.SqliteDatabase(stats_db_path)
  stats_db.connect()

  ScenarioStatistics._meta.database = stats_db
  stats_db.create_tables([ScenarioStatistics])

  origin = cfg.SIMULATION_INSTANCE_NAME

  sync_peewee_table(
      stats_db,
      ScenarioStatistics,
      extra_columns='error',
  )

  with ProcessPoolExecutor(
      max_workers=6, max_tasks_per_child=32,
  ) as executor:
    try:

      futures: Dict[Future[ScenarioStatistics | None], str] = {}
      for s in tqdm(cfg.all_scenarios_grad):
        if stats_exist(s['UniqueName'], origin):
          continue
        f = executor.submit(get_statistics, s, scenario_base_path, origin)
        futures[f] = s['UniqueName']

      n = len(futures)
      print(f'{n} Futures submitted.')

      for f in tqdm(
          as_completed(futures), total=n, bar_format=short_progress_bar,
      ):
        try:
          res = f.result()
          if res is not None:
            res.save()
        except Exception as e:
          print(f"Exception in worker: {e}")
          continue

    except KeyboardInterrupt:
      print("KeyboardInterrupt: terminating all workers...")
      executor.shutdown(wait=False, cancel_futures=True)
      raise
