from typing import List
import sys

from utils.sqlalchemy import create_db_session
from works.stat.types import ScenarioStatistics

if __name__ == "__main__":
  stats_db_path = sys.argv[1]
  flawed_path = sys.argv[2]

  # SQLite数据库连接
  session = create_db_session(stats_db_path)

  # 读取flawed name列表
  with open(flawed_path, 'r', encoding="utf-8") as f:
    flawed_name_list = [x.strip() for x in f.readlines()]

  for flawed_name in flawed_name_list:
    print(flawed_name)
    # 使用like实现contains效果，%%用于转义%字符
    ret: List[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
        ScenarioStatistics.name.like(f"%{flawed_name}%")
    ).all()
    for s in ret:
      session.delete(s)
      print('Deleted:', s.name)
    session.commit()  # 每个flawed_name都commit一次，或移到循环外批量commit

  session.close()
