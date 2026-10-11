"""Execute matched population tasks with the shared engine population recording policy."""

import json
from pathlib import Path

from common.io.json import atomic_json, sha
from common.runtime.process import run_jsonl
from theory.parameters import Parameters
from theory.runtime import measure as schema
from theory.runtime.measure import measure_request
from theory.runtime.protocol import POPULATION_POLICIES, read_response, task_line


def execute(tasks, directory, binary, *, force=False, limit=None):
    """Run or resume exact task-matching JSON caches, using split initial RNG streams and actual graph projection; limit truncates execution, not the design."""
    from smp_meso_bindings.kinetic_runner import prepare_kinetic_request

    directory.mkdir(parents=True, exist_ok=True)
    binding = {"binary_sha256": sha(binary), "request_schema_sha256": sha(Path(schema.__file__))}
    binding_path = directory.parent / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("population binary changed; use a new output directory")
    atomic_json(binding_path, binding)
    for index, task in enumerate(tasks[:limit] if limit else tasks):
        path = directory / f"{task['id']}.json"
        if path.exists() and not force:
            if json.loads(path.read_text())["task"] != task:
                raise ValueError(f"cache protocol differs: {path}")
            print(f"cached {task['id']}", flush=True)
            continue
        raw_request = measure_request(task["id"], Parameters(**task["parameters"]), **task.get("request_options", {}))
        snapshot_steps = sorted({*raw_request["snapshots"]["record_steps"], raw_request["steps"]})
        request = prepare_kinetic_request(raw_request)
        payload = {
            "id": task["id"],
            "kind": "micro",
            "request": request,
            "init_seed": task["initial_seed"],
            "process_seed": task["process_seed"],
            "selection": task["selection"],
            "network": "fixed",
            "physical_dt": 1,
            "snapshot_steps": snapshot_steps,
            "budgets": False,
            "microscopic_initial": {
                "type": "uniform",
                "minimum": -1.0,
                "maximum": 1.0,
                "clip_minimum": -1.0,
                "clip_maximum": 1.0,
                "components": [],
            },
            "micro_options": {"post_retain_count": 1, "repost_rate": 0.0, "recommendation_noise_std": 0.0},
            **POPULATION_POLICIES,
        }
        print(f"run {index + 1}/{len(tasks)} {task['id']}", flush=True)
        result = read_response(run_jsonl(binary, task_line(payload)), task["id"])
        diagnostics = result["pair"]["diagnostics"]
        if max(diagnostics["max_node_mass_residual"], diagnostics["max_fixed_degree_residual"]) > 1e-8:
            raise RuntimeError(f"{task['id']}: pair conservation failed")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"task": task, "result": result}, separators=(",", ":")) + "\n")
        temporary.replace(path)
