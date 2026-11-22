"""Independent implementation-comparison protocol; frozen wire plans stay unchanged."""

from itertools import product

from ehk.metrics.pathway import EXCESS_THRESHOLD, OBSERVATION_INTERVAL, SEPARATION_THRESHOLD
from ehk.micro.scenarios import create_scenario_metadata

STUDY = "recommender_variants"
DESIGN_VERSION = 3
SEED_NAMESPACE = 2  # Preserve the historical namespace independently of registry order.
REFERENCE = "OpinionM9"  # Shared initialization, not a treatment-policy restriction.
REPLICATES = 16

SETTING_PARAMS = [0.01, 0.03, 0.05, 0.10, 0.30]
CROSS_SETTINGS = [(0.01, 0.10), (0.03, 0.30), (0.10, 0.01), (0.30, 0.03)]
# Appending preserves the parameter coordinates used by existing RNG seeds.
SETTINGS = [*zip(SETTING_PARAMS, reversed(SETTING_PARAMS)), *CROSS_SETTINGS]
REPOST_RATES = (0.0, 0.1, 0.25)
CORE_EPSILON = 0.45
CORE_HISTORY = 0
SENSITIVITY_SETTING = (0.05, 0.05)
SENSITIVITY_REPOST = 0.1
SENSITIVITIES = (
    ("epsilon", 0.35, CORE_HISTORY),
    ("epsilon", 0.55, CORE_HISTORY),
    ("history", CORE_EPSILON, 2),
    ("history", CORE_EPSILON, 6),
)

REFERENCE_HORIZON = 2048
VARIANTS = {
    "Opinion": ("Opinion", "OpinionM9", "OpinionRandom"),
    "Structure": ("Structure", "StructureM9", "StructureRandom"),
}


def variant_arms(protocol=None) -> dict[str, dict]:
    """Return arm configurations from a frozen protocol, or the current design.

    The family comparison and the optional factorial extension are represented
    separately in the protocol. Existing family arms inherit task parameters;
    additional arms override only the factors they test.
    """
    if protocol is not None and "variant_arms" in protocol:
        return protocol["variant_arms"]
    families = VARIANTS if protocol is None else protocol["variants"]
    arms = {
        policy: {"recommender": policy, "parameters": {}}
        for family in families.values() for policy in family
    }
    if protocol is None:
        arms.update({
            f"OpinionRandomZeta{zeta}Noise{noise}": {
                "recommender": "OpinionRandom",
                "parameters": {"Steepness": float(zeta), "OpRandomNoiseStd": float(noise)},
            }
            for zeta, noise in ((1, 0), (4, 0), (4, 2))
        })
    return arms

RECOMMENDER_PARAMS = {
    "MixRate": 0.1,
    "NoiseStd": 0.1,
    "OpRandomNoiseStd": 2.0,
    "RandomRatio": 0.0,
    "Tolerance": 0.4,
    "Steepness": 1.0,
    "UseCache": True,
    "LogRecommendations": False,
}
NODE_COUNT = 500
POST_SLOTS = 10


def observation_steps(horizon: int) -> list[int]:
    return [*range(0, horizon, OBSERVATION_INTERVAL), horizon]


def generate_tasks(study: str = STUDY) -> list[dict]:
    if study != STUDY:
        raise ValueError(f"not an implementation-comparison study: {study}")
    seed_namespace = SEED_NAMESPACE
    tasks = []
    coordinates = [
        (setting, repost, "core", CORE_EPSILON, CORE_HISTORY)
        for setting, repost in product(SETTINGS, REPOST_RATES)
    ]
    coordinates.extend(
        (SENSITIVITY_SETTING, SENSITIVITY_REPOST, sensitivity, epsilon, history)
        for sensitivity, epsilon, history in SENSITIVITIES
    )
    for replicate, ((influence, rewiring), repost, sensitivity, epsilon, history) in product(
        range(REPLICATES), coordinates
    ):
        setting = f"al{round(100 * influence):02}_q{round(100 * rewiring):02}_e{round(100 * epsilon):02}_kh{history}"
        name = f"s_{study}_{setting}_p{round(100 * repost):02}_sim{replicate:02}"
        # Sensitivities reuse the matching core initialization, within a study.
        coordinate = SETTINGS.index((influence, rewiring)) * len(REPOST_RATES) + REPOST_RATES.index(repost)
        seed = (seed_namespace << 40) + coordinate * REPLICATES + replicate
        metadata = create_scenario_metadata(
            name,
            influence=influence,
            rewiring=rewiring,
            repost_rate=repost,
            recsys_type=REFERENCE,
            post_retain_count=history,
            tolerance=epsilon,
            recsys_count=POST_SLOTS,
            max_sim_step=REFERENCE_HORIZON,
        )
        metadata.update(
            NodeCount=NODE_COUNT,
            NodeFollowCount=15,
            RecSysParams=dict(RECOMMENDER_PARAMS),
            RNG={
                "Algorithm": "pcg64-dxsm-v1",
                "Seed1": f"0x{0x45484B0000000000 + seed:016x}",
                "Seed2": f"0x{0xB9B7B4FFFFFFFFFF - seed:016x}",
            },
        )
        tasks.append({
            "id": name, "setting": setting, "sensitivity": sensitivity,
            "replicate": replicate, "metadata": metadata,
        })
    return tasks


def design_protocol(study: str = STUDY) -> dict:
    if study != STUDY:
        raise ValueError(f"not an implementation-comparison study: {study}")
    # Legacy campaign fields are retained solely to preserve frozen plan hashes.
    # Actual scheduling is owned by this package.
    common = {
        "replicates": REPLICATES,
        "settings": [list(setting) for setting in SETTINGS],
        "repost_rates": list(REPOST_RATES),
        "core_epsilon": CORE_EPSILON,
        "core_history": CORE_HISTORY,
        "sensitivities": [
            {"axis": axis, "epsilon": epsilon, "history": history,
             "setting": list(SENSITIVITY_SETTING), "repost": SENSITIVITY_REPOST}
            for axis, epsilon, history in SENSITIVITIES
        ],
        "cadence": OBSERVATION_INTERVAL,
        "recording": {
            "backend": "persistent_smp",
            "AgentNumber": True,
            "OpinionSum": True,
            "RewiringEvent": True,
            "PostEvent": True,
            "ViewPostsEvent": False,
            "ViewPostsDetails": False,
            "TrajectoryPrecision": {"opinions": "float64", "opinion_sums": "float64"},
            "prefix": "replay_from_initialization_with_checkpoint_physical_and_rng_equality",
            "late_trace": None,
        },
        "campaign": {
            "order": ["staged", "recommender_variants", "staged_timing"],
            "max_native_workers": 6,
            "timing_requires": "core_stage_gate",
        },
        "exposure": "interval_counters",
        "separation_threshold": SEPARATION_THRESHOLD,
        "excess_threshold": EXCESS_THRESHOLD,
        "cross_camp_margin_in_epsilon": 0.25,
        "reference_horizon": REFERENCE_HORIZON,
        "terminal_major_mass": 0.02,
    }
    return {
        **common, "variants": {family: list(arms) for family, arms in VARIANTS.items()},
        "variant_arms": variant_arms(),
    }
