"""Original-paper task generators; axis order also defines persistent scenario IDs.

Generation is lazy at module import. Each call returns fresh typed metadata;
replicate is the outermost axis and the rightmost axis varies fastest. The mechanism
study is an explicit case list, rather than a Cartesian parameter sweep.
"""

from itertools import product
from dataclasses import dataclass
from ehk.domain import Task, Coordinates

from common.experiments.registry import TaskRegistry
from ehk.micro.scenarios import ScenarioMetadata, create_scenario_metadata, reproducible_metadata

SEED_NAMESPACE = "ehk-paper-recalculation-v1"

INFLUENCE_AXIS = (0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0)
REWIRING_AXIS = (0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0)
REPOST_AXIS = (0.0, 0.1, 0.25, 0.5)
TOLERANCE_AXIS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0)
GRADATION_RECOMMENDATIONS = {
    "st": ("StructureM9", 0),
    "op0": ("OpinionM9", 0),
    "op2": ("OpinionM9", 2),
    "op6": ("OpinionM9", 6),
}
REPLICATE_RECOMMENDATIONS = {"rn": "Random", "st": "StructureM9"}

# Ordered coordinates are part of the historical naming/iteration contract.
GRADATION_AXES = {
    "replicate": tuple(range(12)),
    "rewiring": REWIRING_AXIS,
    "influence": INFLUENCE_AXIS,
    "repost": REPOST_AXIS,
    "recommendation": tuple(GRADATION_RECOMMENDATIONS),
}
EPSILON_AXES = {"replicate": tuple(range(100)), "tolerance": TOLERANCE_AXIS}
REPLICATE_AXES = {"replicate": tuple(range(100)), "recommendation": tuple(REPLICATE_RECOMMENDATIONS)}

# Explicit physical values make the five phase cases inspectable without indices
# into unrelated scan arrays. Unspecified fields retain create_scenario_metadata defaults.
@dataclass(frozen=True)
class MechanismCase:
    rewiring: float
    influence: float
    repost_rate: float
    recsys_type: str
    post_retain_count: int = 3

    def metadata(self, name: str) -> ScenarioMetadata:
        return create_scenario_metadata(name, rewiring=self.rewiring, influence=self.influence,
                                       repost_rate=self.repost_rate, recsys_type=self.recsys_type,
                                       post_retain_count=self.post_retain_count)


MECHANISM_CASES = {
    "baseline": MechanismCase(rewiring=0.05, influence=0.05, repost_rate=0, recsys_type="Random"),
    "baseline_pbs": MechanismCase(rewiring=0.025, influence=0.05, repost_rate=0, recsys_type="Random"),
    "baseline_sbp": MechanismCase(rewiring=0.025, influence=0.005, repost_rate=0, recsys_type="Random"),
    "influence": MechanismCase(rewiring=0.05, influence=0.1, repost_rate=0, recsys_type="Random"),
    "retweet": MechanismCase(rewiring=0.05, influence=0.05, repost_rate=0.1, recsys_type="Random"),
    "op_recsys": MechanismCase(rewiring=0.05, influence=0.05, repost_rate=0, recsys_type="OpinionM9"),
    "phase1": MechanismCase(rewiring=1.0, influence=0.005, repost_rate=0.5, recsys_type="StructureM9", post_retain_count=0),
    "phase2": MechanismCase(rewiring=1.0, influence=0.005, repost_rate=0.0, recsys_type="StructureM9", post_retain_count=0),
    "phase3": MechanismCase(rewiring=0.1, influence=0.1, repost_rate=0.0, recsys_type="OpinionM9", post_retain_count=6),
    "phase4": MechanismCase(rewiring=0.03, influence=1.0, repost_rate=0.0, recsys_type="OpinionM9", post_retain_count=6),
    "phase5": MechanismCase(rewiring=0.03, influence=1.0, repost_rate=0.0, recsys_type="StructureM9", post_retain_count=0),
}
MECHANISM_AXES = {"case": tuple(MECHANISM_CASES)}
SCENARIO_AXES = {
    "epsilon": EPSILON_AXES,
    "gradation": GRADATION_AXES,
    "mech": MECHANISM_AXES,
    "replicate": REPLICATE_AXES,
}


def gradation_tasks() -> list[ScenarioMetadata]:
    return [
        create_scenario_metadata(
            f"s_grad_sim{replicate}_rw{i_rw}_dc{i_dc}_rt{i_rt}_{recommendation}",
            rewiring=rewiring,
            influence=influence,
            repost_rate=repost,
            recsys_type=GRADATION_RECOMMENDATIONS[recommendation][0],
            post_retain_count=GRADATION_RECOMMENDATIONS[recommendation][1],
        )
        for replicate, (i_rw, rewiring), (i_dc, influence), (i_rt, repost), recommendation in product(
            GRADATION_AXES["replicate"],
            enumerate(REWIRING_AXIS),
            enumerate(INFLUENCE_AXIS),
            enumerate(REPOST_AXIS),
            GRADATION_AXES["recommendation"],
        )
    ]


def epsilon_tasks() -> list[ScenarioMetadata]:
    return [
        create_scenario_metadata(
            f"s_eps_sim{replicate}_to{i_tolerance}",
            rewiring=0.05,
            influence=0.05,
            repost_rate=0.1,
            recsys_type="Random",
            post_retain_count=3,
            tolerance=tolerance,
            max_sim_step=20000,
        )
        for replicate, (i_tolerance, tolerance) in product(EPSILON_AXES["replicate"], enumerate(TOLERANCE_AXIS))
    ]


def replicate_tasks() -> list[ScenarioMetadata]:
    return [
        create_scenario_metadata(
            f"s_rep_sim{replicate}_{key}",
            rewiring=0.05,
            influence=0.05,
            repost_rate=0.1,
            recsys_type=REPLICATE_RECOMMENDATIONS[key],
            post_retain_count=3,
            tolerance=0.45,
            max_sim_step=20000,
        )
        for replicate, key in product(REPLICATE_AXES["replicate"], REPLICATE_AXES["recommendation"])
    ]


def mechanism_tasks() -> list[ScenarioMetadata]:
    return [parameters.metadata(f"s_mech_{name}") for name, parameters in MECHANISM_CASES.items()]


TASKS = TaskRegistry(
    {"epsilon": epsilon_tasks, "gradation": gradation_tasks, "mech": mechanism_tasks, "replicate": replicate_tasks}
)


def generate_tasks(name: str) -> list[ScenarioMetadata]:
    """Return the selected study's original ordered metadata, without shared mutations."""
    return TASKS.select(name)[0]()


@dataclass(frozen=True)
class SeedIdentity:
    namespace: str
    study: str
    coordinates: Coordinates


def _coordinates(name: str, values: tuple) -> Coordinates:
    if name == "gradation":
        replicate, rewiring, influence, repost, recommendation = values
        return Coordinates(replicate=replicate, rewiring=rewiring, influence=influence,
                           repost=repost, recommendation=recommendation)
    if name == "epsilon":
        replicate, tolerance = values
        return Coordinates(replicate=replicate, tolerance=tolerance)
    if name == "replicate":
        replicate, recommendation = values
        return Coordinates(replicate=replicate, recommendation=recommendation)
    if name == "mech":
        return Coordinates(case=values[0])
    raise ValueError(f"unknown scenario mode: {name}")


def execution_tasks(name: str, *, seed_namespace: str = SEED_NAMESPACE) -> list[Task]:
    """New-run metadata with stable independent seeds and explicit axis values.

    Original ``generate_tasks`` retains its scientific design and persistent IDs.
    Use a fresh workspace for seeded reruns.
    """
    if not isinstance(seed_namespace, str) or not seed_namespace.strip():
        raise ValueError("seed_namespace must be a nonempty string")
    axes = SCENARIO_AXES[name]
    tasks = []
    for metadata, values in zip(generate_tasks(name), product(*axes.values()), strict=True):
        coordinates = _coordinates(name, values)
        identity = SeedIdentity(seed_namespace, name, coordinates)
        tasks.append(
            Task(id=metadata.UniqueName, coordinates=coordinates, metadata=reproducible_metadata(metadata, identity))
        )
    return tasks
