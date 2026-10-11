"""Parameter-bound NPZ measure-sweep execution with conservation checks."""

import json
from pathlib import Path

import numpy as np

from common.io.json import atomic_json, sha
from theory.parameters import Parameters
from theory.runtime import measure as schema
from theory.runtime.archives import save_result
from theory.runtime.measure import simulate


def execute(tasks, directory, binary, *, force=False, limit=None):
    """Run or resume parameter-matching sweep caches, reject mass/degree conservation failures and atomically save successful NPZ results."""
    binding = {"binary_sha256": sha(binary), "request_schema_sha256": sha(Path(schema.__file__))}
    binding_path = directory / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("measure binary changed; use a new output directory")
    atomic_json(binding_path, binding)
    for index, item in enumerate(tasks[:limit] if limit else tasks):
        path = directory / "raw" / f"{item['id']}.npz"
        if path.exists() and not force:
            with np.load(path, allow_pickle=False) as archive:
                if json.loads(str(archive["metadata"])) != item:
                    raise ValueError(f"cache protocol differs: {path}; use a new output root or --force")
            print(f"cached {item['id']}", flush=True)
            continue
        print(f"run {index + 1}/{len(tasks)} {item['id']}", flush=True)
        result = simulate(binary, item["id"], Parameters(**item["parameters"]), **item.get("request_options", {}))
        diagnostics = result["diagnostics"]
        if max(diagnostics["max_node_mass_residual"], diagnostics["max_fixed_degree_residual"]) > 1e-8:
            raise RuntimeError(f"mass/degree conservation failed: {item['id']}")
        save_result(path, item, result)
