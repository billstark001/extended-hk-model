import sys
import os


from works.config import all_scenarios_grad
from works.stat.tasks import migrate_from_dict
import works.config as cfg


plot_path = cfg.SIMULATION_STAT_DIR
os.makedirs(plot_path, exist_ok=True)
stats_db_path = os.path.join(plot_path, 'stats.db')

if __name__ == '__main__':
  
  stats_path = sys.argv[2]
  origin = sys.argv[1]
  
  migrate_from_dict(
    stats_db_path,
    stats_path,
    origin,
    all_scenarios_grad,
  )
  