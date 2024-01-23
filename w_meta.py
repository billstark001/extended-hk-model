import os
import pickle

root_path = './run/'
scenario_record_path = root_path + 'scenario.pkl'
step_record_path = root_path + 'step_record.pkl'
step_data_path_format = root_path + 'step_data_{step}.pkl'
step_stats_path_format = root_path + 'step_stats_{step}.pkl'

data_interval = 1
stat_interval = 15
total_step = 800

def init():
  os.makedirs(root_path, exist_ok=True)
  
def save_scenario_record(s, force=False):
  if not force and os.path.isfile(scenario_record_path):
    raise RuntimeError('Scenario file exists. Remove it manually if a re-init is desired.')
  with open(scenario_record_path, 'wb') as f:
    pickle.dump(s, f)
    
def load_scenario_record():
  with open(scenario_record_path, 'rb') as f:
    return pickle.load(f)
  
def load_step_record():
  with open(step_record_path, 'rb') as f:
    return pickle.load(f)
  
def update_step_record(current: int):
  with open(step_record_path, 'wb') as f:
    pickle.dump(current, f)
  
def check_status():
  if not os.path.isfile(scenario_record_path):
    return -2
  if not os.path.isfile(step_record_path):
    return -1
  return load_step_record()
  
