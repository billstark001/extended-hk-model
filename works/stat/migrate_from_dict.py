import sys
import os
import json

import peewee
from tqdm import tqdm

from works.config import all_scenarios_grad
from works.stat.task import stats_from_dict, ScenarioStatistics
import works.config as cfg



def stats_exist(name: str, origin: str) -> bool:
  try:
    ScenarioStatistics.get(
        ScenarioStatistics.name == name,
        ScenarioStatistics.origin == origin,
    )
    return True
  except ScenarioStatistics.DoesNotExist:
    return False

# parameters

plot_path = cfg.SIMULATION_PLOT_DIR
os.makedirs(plot_path, exist_ok=True)
stats_db_path = os.path.join(plot_path, 'stats.db')

if __name__ == '__main__':
  
  stats_path = sys.argv[2]
  origin = sys.argv[1]
  
  # db
  stats_db = peewee.SqliteDatabase(stats_db_path)
  stats_db.connect()

  ScenarioStatistics._meta.database = stats_db
  stats_db.create_tables([ScenarioStatistics])
  
  # json
  with open(stats_path, "r", encoding="utf-8") as f:
    stats = json.load(f)
  
  all_s_map = { x['UniqueName']: x for x in all_scenarios_grad }
  
  # migrate
  for stat in tqdm(stats):
    sm = all_s_map[stat['name']]
    if stats_exist(stat["name"], origin):
      continue
    
    try:
      obj = stats_from_dict(sm, stat, origin)
      obj.save()
    except Exception as e:
      print(e)
    
    