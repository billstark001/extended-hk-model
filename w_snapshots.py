import os
import pickle
from datetime import datetime, timezone

DEFAULT_ROOT_PATH = './run/'
DEFAULT_MAX_SNAPSHOTS = 4


def init(root_path=DEFAULT_ROOT_PATH):
  os.makedirs(root_path, exist_ok=True)


def get_snapshot_filename(scenario_name: str):
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  return f"{timestamp}_{scenario_name}_snapshot.pkl"


def save(model_dump, scenario_name: str, root_path=DEFAULT_ROOT_PATH):
  snapshot_filename = get_snapshot_filename(scenario_name)
  snapshot_path = os.path.join(root_path, snapshot_filename)
  with open(snapshot_path, 'wb') as f:
    pickle.dump(model_dump, f)
  return snapshot_filename


def load_latest(scenario_name: str, root_path=DEFAULT_ROOT_PATH):
  snapshots = []
  for filename in os.listdir(root_path):
    if filename.endswith(f"{scenario_name}_snapshot.pkl"):
      snapshots.append(filename)
  if snapshots:
    snapshots.sort()
    for latest_snapshot in snapshots[::-1]:
      snapshot_path = os.path.join(root_path, latest_snapshot)
      try:
        with open(snapshot_path, 'rb') as f:
          ret = pickle.load(f)
        return ret, latest_snapshot
      except:
        continue
  return None, None


def delete_outdated(
    scenario_name: str,
    root_path=DEFAULT_ROOT_PATH,
    max_snapshots: int = DEFAULT_MAX_SNAPSHOTS
):
  snapshots = [
      filename
      for filename in os.listdir(root_path)
      if filename.endswith(f"{scenario_name}_snapshot.pkl")
  ]
  snapshots.sort()
  i = 0
  if len(snapshots) > max_snapshots:
    for old_snapshot in snapshots[:len(snapshots) - max_snapshots]:
      snapshot_path = os.path.join(root_path, old_snapshot)
      os.remove(snapshot_path)
      i += 1
  return i
