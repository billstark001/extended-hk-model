from typing import List

import os
import sys


import peewee


from works.stat.types import ScenarioStatistics, stats_from_dict
import works.config as cfg


if __name__ == "__main__":

  stats_db_path = sys.argv[1]
  flawed_path = sys.argv[2]

  stats_db = peewee.SqliteDatabase(stats_db_path)
  stats_db.connect()

  ScenarioStatistics._meta.database = stats_db

  with open(flawed_path, 'r', encoding="utf-8") as f:
    flawed_name_list = [x.strip() for x in f.readlines()]

  for flawed_name in flawed_name_list:
    print(flawed_name)
    ret: List[ScenarioStatistics] = ScenarioStatistics.select().where(
        ScenarioStatistics.name.contains(flawed_name),  # type: ignore
    )
    for s in ret:
      s.delete_instance()
      print('Deleted:', s.name)

  stats_db.close()
