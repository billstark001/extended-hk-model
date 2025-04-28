from typing import Optional, Dict

import os
import re

re_graph = re.compile(r'graph-(\d+).msgpack')

class RawSimulationRecord:

  def __init__(self, base_dir: str, metadata: dict):
    unique_name = metadata['UniqueName']
    full_path = os.path.join(base_dir, unique_name)
    file_list = os.listdir(full_path)
    file_list.sort()
    # finished mark
    self.is_finished = any(x for x in file_list if x.startswith('finished'))
    has_events_db = 'events.db' in file_list
    acc_state_list = [x for x in file_list if x.startswith('acc-state-')]
    graph_list = [x for x in file_list if x.startswith('graph-')]
    
    self.is_sanitized = has_events_db and len(acc_state_list) > 0 and len(graph_list) > 0
    if not self.is_finished or not self.is_sanitized:
      return
    
    self.events_db_path = os.path.join(full_path, 'events.db')
    self.acc_state_path = os.path.join(full_path, acc_state_list[-1])
    self.graph_paths = {}
    for graph_name in graph_list:
      step_index = int(re_graph.match(graph_name).group(1))
      self.graph_paths[step_index] = os.path.join(full_path, graph_name)
      
  
  