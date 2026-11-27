import json
import os
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

import experiments.paper.scenarios as cfg
from common.experiments.registry import TaskRegistry
from ehk.domain import ScenarioMetadata, parse_metadata
from .models import AnalysisOptions, FirstPassage, PlanSummary
from ehk.metrics.pathway import EXCESS_THRESHOLD, OBSERVATION_INTERVAL, SEPARATION_THRESHOLD
from ehk.metrics.protocol import METRIC_REVISION
from ehk.micro.observables import c
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.statistics import analyze_statistics
from experiments.paper.paths import statistics_database
from experiments.paths import SMP_PROBE_BINARY
from experiments.workspaces import instance_name as resolve_instance_name
from experiments.workspaces import workspace_dir

# These 3 lists control which common statistics are present for each mode.
EPS_COMMON_STATS = [
    "step",
    "active_step",
    "active_step_threshold",
    "g_index_mean_active",
    "x_indices",
    "h_index",
    "p_index",
    "g_index",
    "pathway_integral",
    "rho_edge",
    "rho_post",
    "event_count",
    "event_step_mean",
    "triads",
    "x_mean_vars",
    "mean_vars_smpl",
    "last_community_count",
    "last_community_sizes",
    "last_opinion_peak_count",
]

GRAD_COMMON_STATS = [
    "step",
    "active_step",
    "active_step_threshold",
    "g_index_mean_active",
    "x_indices",
    "h_index",
    "p_index",
    "g_index",
    "pathway_integral",
    "rho_edge",
    "rho_post",
    "event_count",
    "event_step_mean",
    "triads",
    "x_mean_vars",
    "mean_vars_smpl",
    "last_opinion_peak_count",
]

REP_COMMON_STATS = [
    "step",
    "active_step",
    "active_step_threshold",
    "g_index_mean_active",
    "x_indices",
    "h_index",
    "p_index",
    "g_index",
    "pathway_integral",
    "rho_edge",
    "rho_post",
    "event_count",
    "event_step_mean",
    "triads",
    "x_mean_vars",
    "mean_vars_smpl",
    "last_community_count",
    "last_community_sizes",
    "last_opinion_peak_count",
]


def _skip_if_peak_exists(exist_stats: ScenarioStatistics | None) -> bool:
    return _skip_gradation(exist_stats) and exist_stats.last_opinion_peak_count is not None


def _skip_gradation(exist_stats: ScenarioStatistics | None) -> bool:
    return (
        exist_stats is not None
        and exist_stats.complete
        and exist_stats.metric_revision == METRIC_REVISION
        and exist_stats.pathway_integral is not None
        and exist_stats.x_indices is not None
    )


@dataclass(frozen=True)
class ModeConfig:
    name: str
    scenarios: Callable[[], list[cfg.ScenarioMetadata]]
    common_stats: list[str]
    opinion_peak_distance: int
    skip_exist: Callable[[ScenarioStatistics | None], bool]
    include_backdrop: bool


MODE_CONFIGS: dict[str, ModeConfig] = {
    "epsilon": ModeConfig(
        name="epsilon",
        scenarios=cfg.epsilon_tasks,
        common_stats=EPS_COMMON_STATS,
        opinion_peak_distance=20,
        skip_exist=_skip_if_peak_exists,
        include_backdrop=False,
    ),
    "gradation": ModeConfig(
        name="gradation",
        scenarios=cfg.gradation_tasks,
        common_stats=GRAD_COMMON_STATS,
        opinion_peak_distance=50,
        skip_exist=_skip_gradation,
        include_backdrop=True,
    ),
    "replicate": ModeConfig(
        name="replicate",
        scenarios=cfg.replicate_tasks,
        common_stats=REP_COMMON_STATS,
        opinion_peak_distance=50,
        skip_exist=_skip_if_peak_exists,
        include_backdrop=False,
    ),
}


def _populate_statistics(record: ScenarioStatistics, config: ModeConfig, include_probe: bool) -> None:
    selected = set(config.common_stats)
    if "step" in selected:
        record.step = c.total_steps
    if "active_step" in selected:
        record.active_step = c.active_step
    if "active_step_threshold" in selected:
        record.active_step_threshold = c.active_step_threshold
    if "g_index_mean_active" in selected:
        record.g_index_mean_active = c.g_index_mean_active
    if "x_indices" in selected:
        record.x_indices = c.x_indices
    if "h_index" in selected:
        record.h_index = c.h_index
    if "p_index" in selected:
        record.p_index = c.p_index
    if "g_index" in selected:
        record.g_index = c.g_index
    if "pathway_integral" in selected:
        record.pathway_integral = c.gradation_index_hp
    if "rho_edge" in selected:
        record.rho_edge = c.rho_edge
    if "rho_post" in selected:
        record.rho_post = c.rho_post
    if "event_count" in selected:
        record.event_count = c.event_step.size
    if "event_step_mean" in selected:
        record.event_step_mean = float(np.mean(c.event_step)) if c.event_step.size else None
    if "triads" in selected:
        record.triads = c.n_triads
    if "x_mean_vars" in selected:
        record.x_mean_vars = c.x_mean_vars
    if "mean_vars_smpl" in selected:
        record.mean_vars_smpl = c.mean_vars_smpl
    if "last_community_count" in selected:
        record.last_community_count = c.last_community_count
    if "last_community_sizes" in selected:
        record.last_community_sizes = c.last_community_sizes
    if "last_opinion_peak_count" in selected:
        record.last_opinion_peak_count = c.last_opinion_peak_count
    if config.include_backdrop:
        record.p_backdrop, record.h_backdrop, record.g_backdrop = c.p_backdrop, c.h_backdrop, c.g_backdrop
        record.opinion_diff_seg_mean, record.opinion_diff_seg_std = c.opinion_diff_seg_mean, c.opinion_diff_seg_std
    if include_probe:
        record.pathway_label, record.separation_step, record.segregation_step = c.pathway_label, c.separation_step, c.segregation_step


def get_statistics_for_mode(
    scenario_metadata: ScenarioMetadata,
    scenario_base_path: str,
    origin: str,
    exist_stats: ScenarioStatistics | None,
    mode: str,
    active_threshold=0.98,
    min_inactive_value=0.75,
    include_pathway_probe=False,
    endpoint_complement=True,
    metadata_source="record",
):
    from smp_bindings import RawSimulationRecord

    config = MODE_CONFIGS[mode]
    scenario_name = scenario_metadata.UniqueName

    if not os.path.isdir(os.path.join(scenario_base_path, scenario_name)):
        return None
    metadata_path = Path(scenario_base_path) / scenario_name / "metadata.json"
    if metadata_source == "record":
        actual_metadata = parse_metadata(json.loads(metadata_path.read_text()))
        recorded_metadata = actual_metadata
    elif metadata_source == "original-design":
        if metadata_path.is_file():
            raise ValueError(f"persisted metadata exists; select --metadata-source record: {scenario_name}")
        # This is an explicit declaration of the historical experiment design,
        # never a claim that missing parameters or RNG seeds were recorded.
        actual_metadata = replace(scenario_metadata, AgentNumber=True, OpinionSum=True, RewiringEvent=True, PostEvent=True)
        recorded_metadata = None
    else:
        raise ValueError(f"unknown metadata source: {metadata_source}")
    if actual_metadata.UniqueName != scenario_name:
        raise ValueError(f"raw metadata scenario name differs: {scenario_name}")
    for key in ("HKParams", "NodeCount", "RecsysFactoryType", "PostRetainCount", "RecsysCount", "RecSysParams"):
        if (getattr(actual_metadata, key) is not None and getattr(scenario_metadata, key) is not None
                and getattr(actual_metadata, key) != getattr(scenario_metadata, key)):
            raise ValueError(f"raw metadata differs from requested {key}: {scenario_name}")
    scenario_record = RawSimulationRecord(scenario_base_path, actual_metadata)
    if not scenario_record.is_finished:
        return None
    if not scenario_record.is_sanitized:
        raise ValueError(f"raw record is missing required channels for {scenario_name}")

    with scenario_record:
        if actual_metadata.NodeCount is not None and actual_metadata.NodeCount != scenario_record.agents:
            raise ValueError(f"raw population differs from metadata: {scenario_name}")
        actual_metadata = replace(actual_metadata, NodeCount=scenario_record.agents)
        scenario_record.metadata = actual_metadata
        c.set_state(
            scenario_record=scenario_record,
            active_threshold=active_threshold,
            min_inactive_value=min_inactive_value,
            opinion_peak_distance=config.opinion_peak_distance,
            endpoint_complement=endpoint_complement,
        )
        if include_pathway_probe:
            c.set_state(probe_binary=SMP_PROBE_BINARY.resolve())

        hk_params = actual_metadata.HKParams
        record = ScenarioStatistics(
            name=scenario_name,
            origin=origin,
            metadata=actual_metadata,
            recorded_metadata=recorded_metadata,
            metadata_source=metadata_source,
            metadata_provenance=(
                "recorded metadata.json" if recorded_metadata is not None
                else "explicit original-paper design; parameters and collection settings not independently recorded in raw"
            ),
            requested_metadata=scenario_metadata,
            endpoint_complement=endpoint_complement,
            rng_provenance="recorded"
            if actual_metadata.RNG is not None or actual_metadata.RNGSeed is not None
            else "unavailable in raw metadata",
            tolerance=hk_params.Tolerance,
            decay=hk_params.Influence,
            rewiring=hk_params.RewiringRate,
            repost_rate=hk_params.RepostRate,
            recsys_type=actual_metadata.RecsysFactoryType,
            post_retain_count=actual_metadata.PostRetainCount,
        )
        _populate_statistics(record, config, include_pathway_probe)
        return record


STATISTICS = TaskRegistry(MODE_CONFIGS)


def get_mode_names() -> list[str]:
    return sorted(STATISTICS.names)


def run_mode(
    mode: str,
    instance_name: str | None,
    concurrency: int,
    *,
    database=None,
    limit=None,
    force=False,
    include_pathway_probe=False,
    plan_only=False,
    raw_root=None,
    origin=None,
    endpoint_complement=True,
    task_names=None,
    metadata_source="record",
):
    if mode not in MODE_CONFIGS:
        valid = ", ".join(get_mode_names())
        raise ValueError(f"Unsupported mode: {mode}. expected one of: {valid}")

    stats_db_path = str(database or statistics_database())

    if raw_root is None and instance_name is None:
        raise ValueError("provide an instance_name or --raw-root")
    scenario_base_path = str(Path(raw_root).resolve() if raw_root else workspace_dir(instance_name))
    if not os.path.isdir(scenario_base_path) and not plan_only:
        raise FileNotFoundError(scenario_base_path)

    c.set_state(active_threshold=0.98, min_inactive_value=0.75, endpoint_complement=endpoint_complement)

    mode_config = STATISTICS.select(mode)[0]
    get_statistics = partial(
        get_statistics_for_mode,
        mode=mode,
        include_pathway_probe=include_pathway_probe,
        endpoint_complement=endpoint_complement,
        metadata_source=metadata_source,
    )
    scenarios = mode_config.scenarios()
    if task_names is not None:
        requested = set(task_names)
        available = {metadata.UniqueName for metadata in scenarios}
        if not requested or requested - available:
            raise ValueError(f"unknown or empty task selection: {sorted(requested - available)}")
        scenarios = [metadata for metadata in scenarios if metadata.UniqueName in requested]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        scenarios = scenarios[:limit]
    if plan_only:
        return PlanSummary(mode, stats_db_path, scenario_base_path, len(scenarios),
                           os.path.isdir(scenario_base_path), METRIC_REVISION,
                           include_pathway_probe, endpoint_complement, metadata_source, tuple(scenarios))

    return analyze_statistics(
        get_statistics,
        scenario_base_path,
        stats_db_path,
        origin or (str(Path(scenario_base_path).resolve()) if raw_root else resolve_instance_name(instance_name)),
        scenarios,
        mode=mode,
        protocol=AnalysisOptions(
            common_statistics=tuple(mode_config.common_stats), metric_revision=METRIC_REVISION,
            endpoint_complement=endpoint_complement, metadata_source=metadata_source,
            first_passage=FirstPassage(OBSERVATION_INTERVAL, SEPARATION_THRESHOLD, EXCESS_THRESHOLD)
            if include_pathway_probe else None,
            opinion_peak_distance=mode_config.opinion_peak_distance, active_threshold=.98,
            min_inactive_value=.75, extra_statistics="backdrop" if mode_config.include_backdrop else "none",
            skip_existing=mode_config.skip_exist.__name__,
        ),
        ignore_exist=False,
        is_complete=mode_config.skip_exist,
        concurrency=concurrency,
        force=force,
    )
