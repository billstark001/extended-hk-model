"""Cost-evidence I/O; operational metadata reads never replace scientific validation."""

from pathlib import Path

from common.experiments.plans import read_plan
from common.io.records import read_record
from lifted.runtime.costs import measurement
from lifted.runtime.records import record_path


def read_cost_metadata(root, task):
    path = record_path(root, task["id"])
    if not path.exists():
        return None
    record = read_record(path)
    if record["task"] != task:
        raise ValueError("cost record differs from frozen task")
    return record["result"]


def load_pilot_measurements(directory):
    if directory is None or not (Path(directory) / "plan.json").exists():
        return []
    plan = read_plan(Path(directory) / "plan.json")
    rows = []
    for task in plan["tasks"]:
        if task["kind"].endswith("moments"):
            continue  # Progress does not load large increment arrays just to estimate cost.
        result = read_cost_metadata(directory, task)
        if result is not None:
            rows.append(measurement(task, result))
    return rows
