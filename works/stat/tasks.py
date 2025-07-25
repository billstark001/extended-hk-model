from typing import Dict, List, Callable, TypeAlias

import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

from tqdm import tqdm


from utils.sqlalchemy import create_db_engine_and_session, create_db_session, sync_sqlite_table
from works.config import GoMetadataDict, STAT_THREAD_COUNT
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
    [GoMetadataDict, str, str, ScenarioStatistics | None],
    ScenarioStatistics | None,
]


def migrate_from_dict(
    stats_db_path: str,
    stats_path: str,
    origin: str,
    scenarios: List[GoMetadataDict]
):

  stats_session = create_db_session(stats_db_path, ScenarioStatistics.Base)

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
      stats_session.add(obj)
      stats_session.commit()
    except Exception as e:
      print(e)


def generate_stats(
    get_statistics: StatisticsGetterFunc,
    scenario_base_path: str,
    stats_db_path: str,
    origin: str,
    scenarios: List[GoMetadataDict],
    ignore_exist: bool = True,
):

  logger = multiprocessing.util.log_to_stderr()
  logger.setLevel('INFO')

  stats_engine, stats_session = create_db_engine_and_session(
      stats_db_path,
      ScenarioStatistics.Base,
  )

  sync_sqlite_table(
      stats_engine,
      ScenarioStatistics,
      extra_columns='error',
  )

  with ProcessPoolExecutor(
      max_workers=STAT_THREAD_COUNT,
      # max_tasks_per_child=32,
  ) as executor:
    try:

      futures: Dict[Future[ScenarioStatistics | None], str] = {}
      for s in tqdm(scenarios):
        exist_stats = try_get_stats(s['UniqueName'], origin)
        if ignore_exist and exist_stats is not None:
          continue
        # else, upsert
        f = executor.submit(get_statistics, s, scenario_base_path, origin, exist_stats)
        futures[f] = s['UniqueName']

      n = len(futures)
      print(f'{n} Futures submitted.')

      for f in tqdm(
          as_completed(futures), total=n, bar_format=short_progress_bar,
      ):
        try:
          res = f.result()
          if res is not None:
            stats_session.merge(res) # this upserts the record
            stats_session.commit()
        except KeyboardInterrupt:
          print("KeyboardInterrupt: shutting down executor")
          executor.shutdown(wait=False, cancel_futures=True)
          raise  # again
        except Exception as e:
          print(f"Exception in worker: {e}")
          print("Traceback:")
          print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))

          continue

    except KeyboardInterrupt:
      print("KeyboardInterrupt: terminating all workers...")
      executor.shutdown(wait=False, cancel_futures=True)
      raise
