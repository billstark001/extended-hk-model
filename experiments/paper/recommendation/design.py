"""Fixed designs for the paper's paired recommendation experiments."""

from itertools import product

from ehk.metrics.pathway import EXCESS_THRESHOLD, OBSERVATION_INTERVAL, SEPARATION_THRESHOLD
from ehk.micro.scenarios import create_scenario_metadata

STUDIES = ("staged", "staged_timing")
SEED_NAMESPACES = {"staged": 0, "staged_timing": 1}
DESIGN_VERSION = 3
REFERENCE = "OpinionM9"
SWITCH = "StructureM9"
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
FOLLOWUP = 512
FIXED_TRIGGER = 128
LATEST_TRIGGER = 1536
STAGE_CHECKPOINTS = (0, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, REFERENCE_HORIZON)
STAGES = ("early", "middle", "late")
LATE_ACTIVITY = {"opinion_shift": 1e-3, "rewires": 1e-2, "earliest_step": 128, "intervals": 2}
MIN_STAGE_EFFECT = 0.05


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


def generate_tasks(study: str) -> list[dict]:
    if study not in STUDIES:
        raise ValueError(f"unknown intervention study: {study}")
    seed_namespace = SEED_NAMESPACES[study]
    tasks = []
    coordinates = [
        (setting, repost, "core", CORE_EPSILON, CORE_HISTORY)
        for setting, repost in product(SETTINGS, REPOST_RATES)
    ]
    if study != "staged_timing":
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
            max_sim_step=REFERENCE_HORIZON + FOLLOWUP if study == "staged" else REFERENCE_HORIZON,
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


def design_protocol(study: str) -> dict:
    if study not in STUDIES:
        raise ValueError(f"unknown intervention study: {study}")
    # Preserve legacy campaign metadata in frozen plans; it no longer schedules variants.
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
        ] if study != "staged_timing" else [],
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
            "late_trace": "followup_feed_details_jsonl_gzip" if study == "staged" else None,
        },
        "campaign": {
            "order": ["staged", "recommender_variants", "staged_timing"],
            "max_native_workers": 6,
            "timing_requires": "core_stage_gate",
        },
        "exposure": "interval_counters_and_late_feed_trace" if study == "staged" else "interval_counters",
        "separation_threshold": SEPARATION_THRESHOLD,
        "excess_threshold": EXCESS_THRESHOLD,
        "cross_camp_margin_in_epsilon": 0.25,
        "reference_horizon": REFERENCE_HORIZON,
        "terminal_major_mass": 0.02,
    }
    return {
        **common,
        "reference": REFERENCE,
        "switch": SWITCH,
        "window": FOLLOWUP,
        **(
            {
                "checkpoints": list(STAGE_CHECKPOINTS),
                "late_activity": LATE_ACTIVITY,
                "min_stage_effect": MIN_STAGE_EFFECT,
            }
            if study == "staged"
            else {
                "fixed_trigger": FIXED_TRIGGER,
                "latest_trigger": LATEST_TRIGGER,
                "primary_outcome": {
                    "metric": "end_separation",
                    "direction": "lower",
                    "comparison": "state minus fixed",
                },
                "gate_evidence": "completed_core_staged_tasks",
            }
        ),
    }


def select_stages(points: list[dict]) -> dict[str, dict]:
    """Select the first eligible checkpoint of each stage along the reference."""
    selected = {}
    quiet = 0
    for point in points:
        step, state, activity = point["step"], point["pathway"], point["exposure"]
        updates = activity["updates"]
        if (
            updates
            and activity["absolute_shift"] / updates <= LATE_ACTIVITY["opinion_shift"]
            and activity["rewires"] / updates <= LATE_ACTIVITY["rewires"]
        ):
            quiet += 1
        else:
            quiet = 0
        if step not in STAGE_CHECKPOINTS:
            continue
        elevated = state["separation"] >= SEPARATION_THRESHOLD or (
            state["excess_concordance"] is not None and state["excess_concordance"] >= EXCESS_THRESHOLD
        )
        if quiet >= LATE_ACTIVITY["intervals"] and step >= LATE_ACTIVITY["earliest_step"]:
            selected["late"] = point
            break
        if elevated and "early" in selected:
            selected.setdefault("middle", point)
        elif not elevated and "middle" not in selected:
            selected.setdefault("early", point)
    return selected
