import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

import experiments.paper.scenarios as cfg
from common.experiments.registry import TaskRegistry
from ehk.metrics.pathway import EXCESS_THRESHOLD, OBSERVATION_INTERVAL, SEPARATION_THRESHOLD
from ehk.metrics.protocol import METRIC_REVISION
from ehk.micro.observables import c
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.statistics import analyze_statistics
from experiments.paper.paths import STAT_INPUT, statistics_database
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


def _extract_hk_params(scenario_metadata: "cfg.ScenarioMetadata") -> dict:
    return scenario_metadata["HKParams"]


def _skip_if_peak_exists(exist_stats: ScenarioStatistics | None) -> bool:
    return _skip_gradation(exist_stats) and exist_stats.last_opinion_peak_count is not None


def _skip_gradation(exist_stats: ScenarioStatistics | None) -> bool:
    return (
        exist_stats is not None
        and getattr(exist_stats, "complete", False)
        and getattr(exist_stats, "metric_revision", None) == METRIC_REVISION
        and exist_stats.pathway_integral is not None
        and exist_stats.x_indices is not None
    )


def _extra_stats_none() -> dict[str, object]:
    return {}


def _extra_stats_gradation() -> dict[str, object]:
    return {
        "p_backdrop": c.p_backdrop,
        "h_backdrop": c.h_backdrop,
        "g_backdrop": c.g_backdrop,
        "opinion_diff_seg_mean": c.opinion_diff_seg_mean,
        "opinion_diff_seg_std": c.opinion_diff_seg_std,
    }


COMMON_STAT_BUILDERS: dict[str, Callable[[], object]] = {
    "step": lambda: c.total_steps,
    "active_step": lambda: c.active_step,
    "active_step_threshold": lambda: c.active_step_threshold,
    "g_index_mean_active": lambda: c.g_index_mean_active,
    "x_indices": lambda: c.x_indices,
    "h_index": lambda: c.h_index,
    "p_index": lambda: c.p_index,
    "g_index": lambda: c.g_index,
    "pathway_integral": lambda: c.gradation_index_hp,
    "rho_edge": lambda: c.rho_edge,
    "rho_post": lambda: c.rho_post,
    "pathway_label": lambda: c.pathway_label,
    "separation_step": lambda: c.separation_step,
    "segregation_step": lambda: c.segregation_step,
    "event_count": lambda: c.event_step.size,
    "event_step_mean": lambda: float(np.mean(c.event_step)) if c.event_step.size else None,
    "triads": lambda: c.n_triads,
    "x_mean_vars": lambda: c.x_mean_vars,
    "mean_vars_smpl": lambda: c.mean_vars_smpl,
    "last_community_count": lambda: c.last_community_count,
    "last_community_sizes": lambda: c.last_community_sizes,
    "last_opinion_peak_count": lambda: c.last_opinion_peak_count,
}


@dataclass(frozen=True)
class ModeConfig:
    name: str
    scenarios: Callable[[], list[cfg.ScenarioMetadata]]
    common_stats: list[str]
    opinion_peak_distance: int
    skip_exist: Callable[[ScenarioStatistics | None], bool]
    extra_stats_builder: Callable[[], dict[str, object]]


MODE_CONFIGS: dict[str, ModeConfig] = {
    "epsilon": ModeConfig(
        name="epsilon",
        scenarios=cfg.epsilon_tasks,
        common_stats=EPS_COMMON_STATS,
        opinion_peak_distance=20,
        skip_exist=_skip_if_peak_exists,
        extra_stats_builder=_extra_stats_none,
    ),
    "gradation": ModeConfig(
        name="gradation",
        scenarios=cfg.gradation_tasks,
        common_stats=GRAD_COMMON_STATS,
        opinion_peak_distance=50,
        skip_exist=_skip_gradation,
        extra_stats_builder=_extra_stats_gradation,
    ),
    "replicate": ModeConfig(
        name="replicate",
        scenarios=cfg.replicate_tasks,
        common_stats=REP_COMMON_STATS,
        opinion_peak_distance=50,
        skip_exist=_skip_if_peak_exists,
        extra_stats_builder=_extra_stats_none,
    ),
}


def _build_common_stats(selected_names: list[str]) -> dict[str, object]:
    stats: dict[str, object] = {}
    for name in selected_names:
        if name not in COMMON_STAT_BUILDERS:
            raise ValueError(f"Unknown common stat name: {name}")
        stats[name] = COMMON_STAT_BUILDERS[name]()
    return stats


def get_statistics_for_mode(
    scenario_metadata: "cfg.ScenarioMetadata",
    scenario_base_path: str,
    origin: str,
    exist_stats: ScenarioStatistics | None,
    mode: str,
    active_threshold=0.98,
    min_inactive_value=0.75,
    include_pathway_probe=False,
    include_metric_audit=True,
):
    from smp_bindings import RawSimulationRecord

    config = MODE_CONFIGS[mode]
    scenario_name = scenario_metadata["UniqueName"]

    if not os.path.isdir(os.path.join(scenario_base_path, scenario_name)):
        return None
    metadata_path = Path(scenario_base_path) / scenario_name / "metadata.json"
    actual_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else scenario_metadata
    if actual_metadata.get("UniqueName") != scenario_name:
        raise ValueError(f"raw metadata scenario name differs: {scenario_name}")
    for key in ("HKParams", "NodeCount", "RecsysFactoryType", "PostRetainCount", "RecsysCount", "RecSysParams"):
        if key in actual_metadata and key in scenario_metadata and actual_metadata[key] != scenario_metadata[key]:
            raise ValueError(f"raw metadata differs from requested {key}: {scenario_name}")
    scenario_record = RawSimulationRecord(scenario_base_path, actual_metadata)
    if not scenario_record.is_finished:
        return None
    if not scenario_record.is_sanitized:
        raise ValueError(f"raw record is missing required channels for {scenario_name}")

    with scenario_record:
        c.set_state(
            scenario_record=scenario_record,
            probe_binary=SMP_PROBE_BINARY.resolve(),
            active_threshold=active_threshold,
            min_inactive_value=min_inactive_value,
            opinion_peak_distance=config.opinion_peak_distance,
        )

        hk_params = _extract_hk_params(actual_metadata)
        selected_common_stats = _build_common_stats(config.common_stats)
        selected_extra_stats = config.extra_stats_builder()
        if include_metric_audit:
            selected_extra_stats.update(c.metric_audit)
        if include_pathway_probe:
            selected_extra_stats.update(
                {
                    "pathway_label": c.pathway_label,
                    "separation_step": c.separation_step,
                    "segregation_step": c.segregation_step,
                }
            )

        return ScenarioStatistics(
            name=scenario_name,
            origin=origin,
            metadata=actual_metadata,
            requested_metadata=scenario_metadata,
            rng_provenance="recorded"
            if actual_metadata.get("RNG") or actual_metadata.get("RNGSeed")
            else "unavailable in raw metadata",
            tolerance=hk_params["Tolerance"],
            decay=hk_params["Influence"],
            rewiring=hk_params["RewiringRate"],
            repost_rate=hk_params["RepostRate"],
            recsys_type=actual_metadata["RecsysFactoryType"],
            post_retain_count=actual_metadata["PostRetainCount"],
            **selected_common_stats,
            **selected_extra_stats,
        )


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
    include_metric_audit=True,
    plan_only=False,
    raw_root=None,
    origin=None,
):
    if mode not in MODE_CONFIGS:
        valid = ", ".join(get_mode_names())
        raise ValueError(f"Unsupported mode: {mode}. expected one of: {valid}")

    plot_path = str(STAT_INPUT.resolve())
    os.makedirs(plot_path, exist_ok=True)
    stats_db_path = str(database or statistics_database())

    if raw_root is None and instance_name is None:
        raise ValueError("provide an instance_name or --raw-root")
    scenario_base_path = str(Path(raw_root).resolve() if raw_root else workspace_dir(instance_name))
    if not os.path.isdir(scenario_base_path) and not plan_only:
        raise FileNotFoundError(scenario_base_path)

    c.set_state(active_threshold=0.98, min_inactive_value=0.75)

    mode_config = STATISTICS.select(mode)[0]
    get_statistics = partial(
        get_statistics_for_mode,
        mode=mode,
        include_pathway_probe=include_pathway_probe,
        include_metric_audit=include_metric_audit,
    )
    scenarios = mode_config.scenarios()
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        scenarios = scenarios[:limit]
    if plan_only:
        return {
            "mode": mode,
            "database": stats_db_path,
            "workspace": scenario_base_path,
            "planned": len(scenarios),
            "workspace_available": os.path.isdir(scenario_base_path),
            "metric_revision": METRIC_REVISION,
            "include_pathway_probe": include_pathway_probe,
            "include_metric_audit": include_metric_audit,
            "tasks": scenarios,
        }

    return analyze_statistics(
        get_statistics,
        scenario_base_path,
        stats_db_path,
        origin or (str(Path(scenario_base_path).resolve()) if raw_root else resolve_instance_name(instance_name)),
        scenarios,
        mode=mode,
        protocol={
            "common_statistics": mode_config.common_stats,
            "metric_revision": METRIC_REVISION,
            "metric_audit": include_metric_audit,
            "first_passage": {
                "cadence": OBSERVATION_INTERVAL,
                "separation": SEPARATION_THRESHOLD,
                "excess_concordance": EXCESS_THRESHOLD,
            }
            if include_pathway_probe
            else None,
            "opinion_peak_distance": mode_config.opinion_peak_distance,
            "active_threshold": 0.98,
            "min_inactive_value": 0.75,
            "extra_statistics": mode_config.extra_stats_builder.__name__,
            "skip_existing": mode_config.skip_exist.__name__,
        },
        ignore_exist=False,
        is_complete=mode_config.skip_exist,
        concurrency=concurrency,
        force=force,
    )
