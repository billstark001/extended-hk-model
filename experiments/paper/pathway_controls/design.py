"""A small independent design; no pathway classification is used to select times."""

from dataclasses import dataclass, replace
from itertools import product

from ehk.micro.scenarios import reproducible_metadata
from ehk.domain import Coordinates, ControlProtocol, Plan, Task, RANKED_TIE_RULE

from ..scenarios import MECHANISM_CASES

CASES = ("baseline_pbs", "baseline_sbp", "baseline", "op_recsys")
ARMS = ("full", "q_zero", "alpha_zero")
SEED_NAMESPACE = "ehk-paper-pathway-controls-v1"
SNAPSHOT_STEPS = (0, 16, 64, 256, 1024, 2048)


@dataclass(frozen=True, slots=True)
class ControlSeed:
    namespace: str
    coordinates: Coordinates


@dataclass(frozen=True, slots=True)
class ControlDesign:
    cases: tuple[str, ...] = CASES
    replicates: int = 12
    horizon: int = 2048
    node_count: int = 500
    follow_count: int = 15
    snapshot_steps: tuple[int, ...] | None = None
    seed_namespace: str = SEED_NAMESPACE

    def plan(self) -> Plan[ControlProtocol]:
        return make_plan(cases=self.cases, replicates=self.replicates, horizon=self.horizon,
                         node_count=self.node_count, follow_count=self.follow_count,
                         snapshot_steps=self.snapshot_steps, seed_namespace=self.seed_namespace)


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
        coordinates = Coordinates(case=case, replicate=replicate)
        metadata = reproducible_metadata(
            replace(MECHANISM_CASES[case].metadata(name), MaxSimulationStep=horizon),
            ControlSeed(seed_namespace, coordinates),
            node_count=node_count,
            follow_count=follow_count,
        )
        metadata = replace(metadata, DataVersion=3, CheckpointSteps=tuple(steps))
        tasks.append(Task(id=name, coordinates=coordinates, metadata=metadata))
    return Plan(schema=1, study="paper_pathway_controls", profile="paired_process_controls",
                protocol=ControlProtocol(cases=cases, replicates=replicates, horizon=horizon,
                    snapshot_steps=tuple(steps), arms=ARMS, seed_namespace=seed_namespace,
                    pairing="same initial graph, opinions, posts and complete RNG state", classification=None,
                    ranked_tie_rule=RANKED_TIE_RULE),
                tasks=tuple(tasks))


def arm_metadata(task, arm):
    if arm not in ARMS:
        raise ValueError(f"unknown process control: {arm}")
    metadata = replace(task.metadata, UniqueName=task.metadata.UniqueName + "_" + arm)
    if arm == "q_zero":
        metadata = replace(metadata, HKParams=replace(metadata.HKParams, RewiringRate=0.))
    elif arm == "alpha_zero":
        metadata = replace(metadata, HKParams=replace(metadata.HKParams, Influence=0.))
    return metadata
