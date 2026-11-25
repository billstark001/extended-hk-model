"""A small independent design; no pathway classification is used to select times."""

from copy import deepcopy
from itertools import product

from ehk.micro.scenarios import create_scenario_metadata, reproducible_metadata

from ..scenarios import MECHANISM_CASES

CASES = ("baseline_pbs", "baseline_sbp", "baseline", "op_recsys")
ARMS = ("full", "q_zero", "alpha_zero")
SEED_NAMESPACE = "ehk-paper-pathway-controls-v1"
SNAPSHOT_STEPS = (0, 16, 64, 256, 1024, 2048)


def make_plan(
    *,
    cases=CASES,
    replicates=12,
    horizon=2048,
    node_count=500,
    follow_count=15,
    snapshot_steps=None,
    seed_namespace=SEED_NAMESPACE,
):
    cases = tuple(cases)
    if not cases or len(set(cases)) != len(cases) or set(cases) - set(CASES):
        raise ValueError("cases must select distinct declared mechanism cases")
    if type(replicates) is not int or replicates < 1 or type(horizon) is not int or horizon < 1:
        raise ValueError("replicates and horizon must be positive integers")
    steps = list(snapshot_steps) if snapshot_steps is not None else [s for s in SNAPSHOT_STEPS if s <= horizon]
    steps = sorted(set([0, horizon, *steps]))
    if any(type(s) is not int or not 0 <= s <= horizon for s in steps):
        raise ValueError("snapshot steps must be integer steps inside the fixed horizon")
    if not isinstance(seed_namespace, str) or not seed_namespace.strip():
        raise ValueError("seed_namespace must be nonempty")
    tasks = []
    for replicate, case in product(range(replicates), cases):
        name = f"s_control_{case}_sim{replicate:02}"
        coordinates = {"case": case, "replicate": replicate}
        metadata = reproducible_metadata(
            create_scenario_metadata(name, max_sim_step=horizon, **MECHANISM_CASES[case]),
            {"namespace": seed_namespace, "coordinates": coordinates},
            node_count=node_count,
            follow_count=follow_count,
        )
        metadata.update(DataVersion=3, CheckpointSteps=steps)
        tasks.append({"id": name, "coordinates": coordinates, "metadata": metadata})
    return {
        "schema": 1,
        "study": "paper_pathway_controls",
        "profile": "paired_process_controls",
        "protocol": {
            "cases": list(cases),
            "replicates": replicates,
            "horizon": horizon,
            "snapshot_steps": steps,
            "arms": list(ARMS),
            "seed_namespace": seed_namespace,
            "pairing": "same initial graph, opinions, posts and complete RNG state",
            "classification": None,
        },
        "tasks": tasks,
    }


def arm_metadata(task, arm):
    if arm not in ARMS:
        raise ValueError(f"unknown process control: {arm}")
    metadata = deepcopy(task["metadata"])
    metadata["UniqueName"] += "_" + arm
    if arm == "q_zero":
        metadata["HKParams"]["RewiringRate"] = 0.0
    elif arm == "alpha_zero":
        metadata["HKParams"]["Influence"] = 0.0
    return metadata
