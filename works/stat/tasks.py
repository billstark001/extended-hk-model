from typing import Dict, List, Callable, TypeAlias

import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

import peewee
from tqdm import tqdm


from utils.peewee import sync_peewee_table
from works.config import GoMetadataDict
from works.stat.types import ScenarioStatistics, stats_from_dict
import multiprocessing.util


short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


def try_get_stats(name: str, origin: str) -> ScenarioStatistics | None:
  try:
    ret = ScenarioStatistics.get(
        ScenarioStatistics.name == name,
        ScenarioStatistics.origin == origin,
    )
    return ret
  except ScenarioStatistics.DoesNotExist:
    return None


StatisticsGetterFunc: TypeAlias = Callable[
    [GoMetadataDict, str, str],
    ScenarioStatistics | None,
]


def migrate_from_dict(
    stats_db_path: str,
    stats_path: str,
    origin: str,
    scenarios: List[GoMetadataDict]
):

  stats_db = peewee.SqliteDatabase(stats_db_path)
  stats_db.connect()

  ScenarioStatistics._meta.database = stats_db
  stats_db.create_tables([ScenarioStatistics])

  # json
  with open(stats_path, "r", encoding="utf-8") as f:
    stats = json.load(f)

  all_s_map = {x['UniqueName']: x for x in scenarios}

  # migrate
  for stat in tqdm(stats):
    sm = all_s_map[stat['name']]
    if try_get_stats(stat["name"], origin) is not None:
      continue

    try:
      obj = stats_from_dict(sm, stat, origin)
      obj.save()
    except Exception as e:
      print(e)


def generate_stats(
    get_statistics: StatisticsGetterFunc,
    scenario_base_path: str,
    stats_db_path: str,
    origin: str,
    scenarios: List[GoMetadataDict],
):

  logger = multiprocessing.util.log_to_stderr()
  logger.setLevel('INFO')  # 可选: DEBUG/INFO/WARNING/ERROR

  stats_db = peewee.SqliteDatabase(stats_db_path)
  stats_db.connect()

  ScenarioStatistics._meta.database = stats_db
  stats_db.create_tables([ScenarioStatistics])

  sync_peewee_table(
      stats_db,
      ScenarioStatistics,
      extra_columns='error',
  )

  with ProcessPoolExecutor(
      max_workers=6,
      # max_tasks_per_child=32,
  ) as executor:
    try:

      futures: Dict[Future[ScenarioStatistics | None], str] = {}
      for s in tqdm(scenarios):
        exist_stats = try_get_stats(s['UniqueName'], origin)
        if exist_stats is not None:
          # TODO add this to func
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
        except KeyboardInterrupt:
          print("KeyboardInterrupt: shutting down executor")
          executor.shutdown(wait=False, cancel_futures=True)
          raise  # again
        except Exception as e:
          print(f"Exception in worker: {e}")
          continue

    except KeyboardInterrupt:
      print("KeyboardInterrupt: terminating all workers...")
      executor.shutdown(wait=False, cancel_futures=True)
      raise
