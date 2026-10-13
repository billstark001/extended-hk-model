"""Parameter-bound NPZ measure-sweep execution with conservation checks."""

import json
from pathlib import Path

from common.io.json import atomic_json, sha
from common.runtime.failures import ProtocolError, TaskError
from common.runtime.process import run_tasks
from theory.parameters import Parameters
from theory.runtime import measure
from theory.runtime.archives import save_result
from theory.runtime.completed import cached


def execute(tasks, directory, binary, *, force=False, limit=None, workers=1, store=None):
    """Run or resume parameter-matching sweep caches, reject mass/degree conservation failures and atomically save successful NPZ results."""
    binding = {"binary_sha256": sha(binary), "request_schema_sha256": sha(Path(measure.__file__))}
    binding_path = directory / "binding.json"

    def bind():
        if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
            raise ProtocolError("measure binary changed; use a new output directory")
        atomic_json(binding_path, binding)

    store.read(bind) if store else bind()

    def run(index_item):
        index, item = index_item
        path = directory / "raw" / f"{item['id']}.npz"

        def check():
            return path.exists() and not force and cached(path, item, "measure")

        if store.read(check) if store else check():
            print(f"cached {item['id']}", flush=True)
            return
        print(f"run {index + 1}/{len(tasks)} {item['id']}", flush=True)
        result = measure.run_measure(
            binary, item["id"], Parameters(**item["parameters"]), **item.get("request_options", {})
        )
        diagnostics = result["diagnostics"]
        if max(diagnostics["max_node_mass_residual"], diagnostics["max_fixed_degree_residual"]) > 1e-8:
            raise TaskError(f"mass/degree conservation failed: {item['id']}")
        if store:
            store.commit(path, lambda staged: save_result(staged, item, result))
        else:
            save_result(path, item, result)

    selected = tasks[:limit] if limit else tasks
    run_tasks(run, enumerate(selected), workers, isolate_failures=True, event=store.event if store else None)
