"""Population request contracts, shared pair identity and binary adapter execution."""

import os
import tempfile
from pathlib import Path
from typing import cast

from common.io.records import read_transport
from common.runtime.process import run_binary
from common.runtime.storage import RunStorage
from common.types import PathInput, RuntimeBinding
from theory.parameters import Parameters
from theory.types import AdapterResult, MicroTask, PairIdentity, PopulationRequest

from .measure import measure_request
from .protocol import POPULATION_POLICIES, task_line


def population_request(task: MicroTask) -> PopulationRequest:
    """Bind frozen physical parameters and seeds to the public population adapter."""
    from smp_meso_bindings.kinetic_runner import prepare_kinetic_request

    raw = measure_request(task["id"], Parameters(**task["parameters"]), **task.get("request_options", {}))
    return {
        "id": task["id"],
        "kind": "micro",
        "request": prepare_kinetic_request(raw),
        "init_seed": task["initial_seed"],
        "process_seed": task["process_seed"],
        "selection": task["selection"],
        "network": "fixed",
        "physical_dt": 1,
        "snapshot_steps": sorted({*raw["snapshots"]["record_steps"], raw["steps"]}),
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
        "compact_records": True,
        **POPULATION_POLICIES,
    }


def pair_identity(task: MicroTask, binding: RuntimeBinding) -> PairIdentity:
    """Process seeds/branch labels do not enter deterministic pair identity."""
    identity = {k: v for k, v in task.items() if k not in ("id", "branch", "process_seed", "roles", "kind")}
    return cast(PairIdentity, {**identity, "binding": binding})


def binary_result(
    binary: PathInput, payload: PopulationRequest, store: RunStorage | None, local: PathInput
) -> AdapterResult:
    """Execute a cancellable binary request; cap and remove its local transport file."""
    if store:
        path = store.temporary()
    else:
        fd, filename = tempfile.mkstemp(dir=local, suffix=".transport")
        os.close(fd)
        path = Path(filename)
    try:
        run_binary(
            binary,
            task_line(payload),
            path,
            arguments=("--msgpack",),
            max_bytes=store.max_record_bytes if store else 512 << 20,
        )
        return cast(AdapterResult, read_transport(path, payload["id"]))
    finally:
        path.unlink(missing_ok=True)
