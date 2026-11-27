"""Typed full-trajectory and OpinionRandom factorial comparisons."""

from dataclasses import dataclass
import pandas as pd

from ehk.domain import MeanInterval, rows_frame
from experiments.paper.recommendation_shared import analysis as shared
from . import design as cfg


@dataclass(frozen=True)
class FactorTerm:
    zeta: float
    score_noise: float
    coefficient: float

    @property
    def factors(self) -> tuple[float, float]:
        return self.zeta, self.score_noise


@dataclass(frozen=True)
class FactorialContrast:
    terms: tuple[FactorTerm, ...]


FACTORIAL_CONTRASTS = {
    "zeta_at_score_noise_0": FactorialContrast((FactorTerm(4, 0, 1), FactorTerm(1, 0, -1))),
    "zeta_at_score_noise_2": FactorialContrast((FactorTerm(4, 2, 1), FactorTerm(1, 2, -1))),
    "score_noise_at_zeta_1": FactorialContrast((FactorTerm(1, 2, 1), FactorTerm(1, 0, -1))),
    "score_noise_at_zeta_4": FactorialContrast((FactorTerm(4, 2, 1), FactorTerm(4, 0, -1))),
    "interaction": FactorialContrast((FactorTerm(4, 2, 1), FactorTerm(1, 2, -1),
                                      FactorTerm(4, 0, -1), FactorTerm(1, 0, 1))),
}


@dataclass(frozen=True)
class VariantColumns:
    recommender: str
    zeta: float | None = None
    score_noise: float | None = None


@dataclass(kw_only=True)
class VariantCoverageRow:
    setting: str
    sensitivity: str
    repost: float
    arm: str
    recommender: str
    zeta: float | None
    score_noise: float | None
    binary_sha256: str | None
    planned: int = 0
    completed: int = 0


@dataclass(kw_only=True)
class FactorialSummary(MeanInterval):
    sensitivity: str
    epsilon: float
    history: int
    setting: str
    repost: float | str
    stage: str
    contrast: str
    metric: str


def completed_task_ids(plan, records):
    if plan.profile != cfg.STUDY:
        raise ValueError(f"not an implementation-comparison profile: {plan.profile}")
    expected = set(cfg.variant_arms(plan.protocol))
    return {task.id for task in plan.tasks if task.id in records and expected <= set(records[task.id].arms)}


def _variant_columns(metadata, configuration) -> VariantColumns:
    if configuration.recommender == "OpinionRandom":
        if metadata.RecSysParams is None:
            raise ValueError("OpinionRandom requires recorded recommendation parameters")
        parameters = metadata.RecSysParams.with_overrides(configuration.parameters)
        return VariantColumns(configuration.recommender, parameters.Steepness, parameters.OpRandomNoiseStd)
    return VariantColumns(configuration.recommender)


def read_study(directory, study=None):
    return shared.read_study(directory, study, profiles=(cfg.STUDY,))


def study_tables(plan, records, *, binding=None):
    if plan.profile != cfg.STUDY:
        raise ValueError(f"not an implementation-comparison profile: {plan.profile}")
    configurations = cfg.variant_arms(plan.protocol)
    return shared.study_tables(
        plan, records, binding=binding,
        arm_columns=lambda metadata, name: _variant_columns(metadata, configurations[name]),
    )


def coverage_by_setting(plan, records):
    return shared.coverage_by_setting(plan, records, completed=completed_task_ids(plan, records))


def completed_cell_task_ids(plan, records):
    return shared.completed_cell_task_ids(plan, records, coverage=coverage_by_setting(plan, records))


def contrast_table(runs):
    pairs = [(policy, family[1]) for family in cfg.VARIANTS.values() for policy in (family[0], family[2])]
    pairs.extend(zip(cfg.VARIANTS["Opinion"], cfg.VARIANTS["Structure"]))
    return shared.contrast_table(runs, pairs)


def variant_arm_coverage(plan, records, *, binding=None):
    if plan.profile != cfg.STUDY:
        raise ValueError(f"not an implementation-comparison profile: {plan.profile}")
    groups: dict[tuple[str, float, str], VariantCoverageRow] = {}
    configurations = cfg.variant_arms(plan.protocol)
    for task in plan.tasks:
        repost = task.metadata.HKParams.RepostRate
        available = records[task.id].arms if task.id in records else {}
        for name, configuration in configurations.items():
            key = task.setting, repost, name
            binary_sha256 = shared._arm_binary_sha256(binding, name, available.get(name))
            if key not in groups:
                columns = _variant_columns(task.metadata, configuration)
                groups[key] = VariantCoverageRow(
                    setting=task.setting, sensitivity=task.sensitivity, repost=repost, arm=name,
                    recommender=columns.recommender, zeta=columns.zeta, score_noise=columns.score_noise,
                    binary_sha256=binary_sha256,
                )
            group = groups[key]
            if binary_sha256 is not None:
                if group.binary_sha256 not in (None, binary_sha256):
                    raise ValueError(f"arm uses different binaries across repeats: {name}")
                group.binary_sha256 = binary_sha256
            group.planned += 1
            group.completed += name in available
    return rows_frame(groups.values(), VariantCoverageRow)


def opinion_random_factorial(plan, runs):
    columns = ["sensitivity", "epsilon", "history", "setting", "repost", "stage", "contrast", "metric",
               "n", "mean", "lo", "hi"]
    if plan.profile != cfg.STUDY:
        raise ValueError(f"not an implementation-comparison profile: {plan.profile}")
    planned = set()
    for configuration in cfg.variant_arms(plan.protocol).values():
        if configuration.recommender == "OpinionRandom":
            factors = _variant_columns(plan.tasks[0].metadata, configuration)
            planned.add((factors.zeta, factors.score_noise))
    if not {(1, 0), (1, 2), (4, 0), (4, 2)} <= planned:
        return pd.DataFrame(columns=columns)
    summaries = []
    for identifying, block in shared.analysis_strata(runs):
        for stage, phases in block.groupby("stage"):
            selected = phases[phases["recommender"] == "OpinionRandom"]
            for metric in shared.EFFECTS:
                paired = selected.pivot(index="task_id", columns=["zeta", "score_noise"], values=metric)
                for name, contrast in FACTORIAL_CONTRASTS.items():
                    keys = [term.factors for term in contrast.terms]
                    values = []
                    if set(keys) <= set(paired.columns):
                        observations = paired[keys].astype(float).dropna()
                        values = sum(observations[term.factors] * term.coefficient for term in contrast.terms)
                    estimate = shared.mean_interval(values)
                    summaries.append(FactorialSummary(
                        sensitivity=identifying.sensitivity, epsilon=identifying.epsilon, history=identifying.history,
                        setting=identifying.setting, repost=identifying.repost, stage=stage, contrast=name, metric=metric,
                        n=estimate.n, mean=estimate.mean, lo=estimate.lo, hi=estimate.hi,
                    ))
    return rows_frame(summaries, FactorialSummary).reindex(columns=columns)
