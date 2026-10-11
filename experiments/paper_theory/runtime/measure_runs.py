"""Parameter-bound NPZ measure-sweep execution with conservation checks."""

import json
from pathlib import Path

from common.io.json import atomic_json, sha
from common.runtime.process import run_jsonl, run_tasks
from theory.parameters import Parameters
from theory.runtime import measure as schema
from theory.runtime.archives import save_result
from theory.runtime.measure import measure_request

from .completed import cached


def simulate(binary, identifier, parameters, **options):
    """Use the binding's exact codec with cancellable subprocess transport."""
    from smp_meso_bindings.kinetic_runner import decode_kinetic_response, prepare_kinetic_request

    request = prepare_kinetic_request(measure_request(identifier, parameters, **options))
    response = decode_kinetic_response(json.loads(run_jsonl(binary, json.dumps(request) + "\n", arguments=("batch",))))
    if (
        response.get("request_id") != identifier
        or response.get("error")
        or not isinstance(response.get("result"), dict)
    ):
        raise RuntimeError(f"{identifier}: invalid kinetic response: {response.get('error')}")
    return response["result"]


def execute(tasks, directory, binary, *, force=False, limit=None, workers=1):
    """Run or resume parameter-matching sweep caches, reject mass/degree conservation failures and atomically save successful NPZ results."""
    binding = {"binary_sha256": sha(binary), "request_schema_sha256": sha(Path(schema.__file__))}
    binding_path = directory / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("measure binary changed; use a new output directory")
    atomic_json(binding_path, binding)

    def run(index_item):
        index, item = index_item
        path = directory / "raw" / f"{item['id']}.npz"
        if path.exists() and not force and cached(path, item, "measure"):
            print(f"cached {item['id']}", flush=True)
            return
        print(f"run {index + 1}/{len(tasks)} {item['id']}", flush=True)
        result = simulate(binary, item["id"], Parameters(**item["parameters"]), **item.get("request_options", {}))
        diagnostics = result["diagnostics"]
        if max(diagnostics["max_node_mass_residual"], diagnostics["max_fixed_degree_residual"]) > 1e-8:
            raise RuntimeError(f"mass/degree conservation failed: {item['id']}")
        save_result(path, item, result)

    selected = tasks[:limit] if limit else tasks
    run_tasks(run, enumerate(selected), workers)
