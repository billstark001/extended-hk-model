from typing import Dict, List, Callable, TypeAlias

import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

from tqdm import tqdm
from sqlalchemy.orm import Session, undefer

from utils.context import Context
from utils.sqlalchemy import create_db_engine_and_session, create_db_session, sync_sqlite_table
from works.config import GoMetadataDict, STAT_THREAD_COUNT
from works.stat.types import ScenarioStatistics, stats_from_dict
import multiprocessing.util


def merge_stats_to_context(
    stats: ScenarioStatistics | None,
    ctx: Context,
    ctx_name_to_stat_name: dict[str, str] | None = None,
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
):
  if stats is None:
    return

  sel_names = set(include_names if include_names else ctx.get_state_names())
  if exclude_names:
    sel_names = sel_names - set(exclude_names)

  state_dict = {}
  if ctx_name_to_stat_name is None:
    ctx_name_to_stat_name = {}
  for sel_name in sel_names:
    stat_name = ctx_name_to_stat_name[sel_name] if sel_name in ctx_name_to_stat_name else sel_name
    if hasattr(stats, stat_name):
      value = getattr(stats, stat_name)
      if value is not None:
        state_dict[sel_name] = value
  ctx.set_state(**state_dict)


short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"

class AnyAttr:
    pass

def try_get_stats(sess: Session, name: str, origin: str) -> ScenarioStatistics | None:
  ret = sess.query(ScenarioStatistics).filter(
      ScenarioStatistics.name == name,
      ScenarioStatistics.origin == origin,
  ).options(undefer('*')).first()
  if not ret:
    return None
  ret2 = AnyAttr()
  for column in ScenarioStatistics.__table__.columns:
    attr = getattr(ret, column.name)
    setattr(ret2, column.name, attr)
  return ret2 # type: ignore


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
    if try_get_stats(stats_session, stat["name"], origin) is not None:
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
        exist_stats = try_get_stats(stats_session, s['UniqueName'], origin)
        if ignore_exist and exist_stats is not None:
          continue
        # else, upsert
        f = executor.submit(
            get_statistics, s, scenario_base_path, origin, exist_stats)
        futures[f] = s['UniqueName']

      n = len(futures)
      print(f'{n} Futures submitted.')

      for f in tqdm(
          as_completed(futures), total=n, bar_format=short_progress_bar,
      ):
        f_name = futures[f]
        try:
          res = f.result()
          if res is not None:
            stats_session.merge(res)  # this upserts the record
            stats_session.commit()
        except KeyboardInterrupt:
          print("KeyboardInterrupt: shutting down executor")
          executor.shutdown(wait=False, cancel_futures=True)
          raise  # again
        except Exception as e:
          print(f"Exception in worker: {e}")
          print(f"Scenario: {f_name}")
          print("Traceback:")
          print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))

          continue

    except KeyboardInterrupt:
      print("KeyboardInterrupt: terminating all workers...")
      executor.shutdown(wait=False, cancel_futures=True)
      raise
