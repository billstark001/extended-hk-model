"""Full-trajectory recommendation-family and OpinionRandom factorial comparisons."""

import pandas as pd

from experiments.paper.recommendation_shared import analysis as shared

from . import design as cfg

FACTORIAL_CONTRASTS = {
    "zeta_at_score_noise_0": {(4, 0): 1, (1, 0): -1},
    "zeta_at_score_noise_2": {(4, 2): 1, (1, 2): -1},
    "score_noise_at_zeta_1": {(1, 2): 1, (1, 0): -1},
    "score_noise_at_zeta_4": {(4, 2): 1, (4, 0): -1},
    "interaction": {(4, 2): 1, (1, 2): -1, (4, 0): -1, (1, 0): 1},
}


def completed_task_ids(plan, records):
    """Require every arm specified by this frozen protocol, including legacy six-arm plans."""
    if plan["profile"] != cfg.STUDY:
        raise ValueError(f"not an implementation-comparison profile: {plan['profile']}")
    expected = set(cfg.variant_arms(plan["protocol"]))
    return {
        task["id"] for task in plan["tasks"]
        if task["id"] in records and expected <= set(records[task["id"]]["arms"])
    }


def _variant_columns(metadata, configuration):
    values = {"recommender": configuration["recommender"], "zeta": None, "score_noise": None}
    if configuration["recommender"] == "OpinionRandom":
        parameters = {**metadata["RecSysParams"], **configuration["parameters"]}
        values.update(zeta=parameters["Steepness"], score_noise=parameters["OpRandomNoiseStd"])
    return values


def read_study(directory, study=None):
    return shared.read_study(directory, study, profiles=(cfg.STUDY,))


def study_tables(plan, records, *, binding=None):
    if plan["profile"] != cfg.STUDY:
        raise ValueError(f"not an implementation-comparison profile: {plan['profile']}")
    configurations = cfg.variant_arms(plan["protocol"])
    return shared.study_tables(
        plan, records, binding=binding,
        arm_columns=lambda metadata, name: _variant_columns(metadata, configurations[name]),
    )


def coverage_by_setting(plan, records):
    return shared.coverage_by_setting(plan, records, completed=completed_task_ids(plan, records))


def completed_cell_task_ids(plan, records):
    return shared.completed_cell_task_ids(plan, records, coverage=coverage_by_setting(plan, records))


def contrast_table(runs):
    pairs = [
        (policy, family[1]) for family in cfg.VARIANTS.values() for policy in (family[0], family[2])
    ] + list(zip(cfg.VARIANTS["Opinion"], cfg.VARIANTS["Structure"]))
    return shared.contrast_table(runs, pairs)


def variant_arm_coverage(plan, records, *, binding=None):
    """Keep completed original arms visible while factorial additions are pending."""
    if plan["profile"] != cfg.STUDY:
        raise ValueError(f"not an implementation-comparison profile: {plan['profile']}")
    groups = {}
    configurations = cfg.variant_arms(plan["protocol"])
    for task in plan["tasks"]:
        repost = task["metadata"]["HKParams"]["RepostRate"]
        available = records.get(task["id"], {}).get("arms", {})
        for name, configuration in configurations.items():
            key = task["setting"], repost, name
            binary_sha256 = shared._arm_binary_sha256(binding, name, available.get(name, {}))
            if key not in groups:
                groups[key] = {
                    "setting": task["setting"], "sensitivity": task["sensitivity"], "repost": repost,
                    "arm": name, **_variant_columns(task["metadata"], configuration),
                    "binary_sha256": binary_sha256, "planned": 0, "completed": 0,
                }
            elif binary_sha256 is not None:
                if groups[key]["binary_sha256"] not in (None, binary_sha256):
                    raise ValueError(f"arm uses different binaries across repeats: {name}")
                groups[key]["binary_sha256"] = binary_sha256
            groups[key]["planned"] += 1
            groups[key]["completed"] += name in available
    return pd.DataFrame(groups.values())


def opinion_random_factorial(plan, runs):
    """Pair factor changes within initial states; interaction is the difference of zeta effects."""
    columns = [
        "sensitivity", "epsilon", "history", "setting", "repost", "stage", "contrast", "metric",
        "n", "mean", "lo", "hi",
    ]
    if plan["profile"] != cfg.STUDY:
        raise ValueError(f"not an implementation-comparison profile: {plan['profile']}")
    planned = set()
    for configuration in cfg.variant_arms(plan["protocol"]).values():
        if configuration["recommender"] == "OpinionRandom":
            factors = _variant_columns(plan["tasks"][0]["metadata"], configuration)
            planned.add((factors["zeta"], factors["score_noise"]))
    if not {(1, 0), (1, 2), (4, 0), (4, 2)} <= planned:
        return pd.DataFrame(columns=columns)
    summaries = []
    for identifying, block in shared.analysis_strata(runs):
        for stage, phases in block.groupby("stage"):
            selected = phases[phases["recommender"] == "OpinionRandom"]
            for metric in shared.EFFECTS:
                paired = selected.pivot(index="task_id", columns=["zeta", "score_noise"], values=metric)
                for contrast, coefficients in FACTORIAL_CONTRASTS.items():
                    values = []
                    if set(coefficients) <= set(paired.columns):
                        observations = paired[list(coefficients)].astype(float).dropna()
                        values = sum(observations[key] * coefficient for key, coefficient in coefficients.items())
                    summaries.append({
                        **identifying, "stage": stage, "contrast": contrast, "metric": metric,
                        **shared.mean_interval(values),
                    })
    return pd.DataFrame(summaries, columns=columns)
