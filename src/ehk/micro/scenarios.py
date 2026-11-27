"""Reusable scenario metadata helpers for the external SMP runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from ehk.domain import ScenarioMetadata, HKParams, Rng, RecommenderParams, TrajectoryPrecision

from common.io.json import fingerprint

RECOMMENDER_DEFAULTS = RecommenderParams(NoiseStd=0.1, OpRandomNoiseStd=2.0, UseCache=True,
    Tolerance=0.4, Steepness=1.0, RandomRatio=0.0, MixRate=0.1, LogRecommendations=False)


def reproducible_metadata(
    metadata: ScenarioMetadata, seed_identity: Any, *, node_count: int = 500, follow_count: int = 15
) -> ScenarioMetadata:
    """Resolve new-run defaults without changing historical scenario dictionaries.

    The identity is a frozen experimental coordinate, not a process/thread ID.
    Identical identities deliberately share initialization and process RNG streams.
    Old unseeded records are never assigned these seeds retrospectively.
    """
    if type(node_count) is not int or type(follow_count) is not int or not 1 <= follow_count < node_count:
        raise ValueError("require integer 1 <= follow_count < node_count")
    digest = fingerprint({"rng_protocol": "ehk-independent-pcg-v1", "identity": seed_identity})
    return replace(metadata,
        DataVersion=2, NodeCount=node_count, NodeFollowCount=follow_count, NetworkType="Random",
        RNG=Rng(Algorithm="pcg64-dxsm-v1", Seed1="0x" + digest[:16], Seed2="0x" + digest[16:32]),
        RecSysParams=RECOMMENDER_DEFAULTS.with_overrides(metadata.RecSysParams or RecommenderParams()),
        AgentNumber=True, OpinionSum=True, RewiringEvent=True, PostEvent=True,
        ViewPostsEvent=False, ViewPostsDetails=False,
        TrajectoryPrecision=TrajectoryPrecision(opinions="float64", opinion_sums="float64"))


def create_scenario_metadata(
    name: str,
    tolerance: float = 0.45,
    influence: float = 0.9,
    rewiring: float = 0.01,
    repost_rate: float = 0.05,
    recsys_type: str = "Random",
    recsys_count: int = 10,
    post_retain_count: int = 3,
    max_sim_step: int = 20_000,
) -> ScenarioMetadata:
    return ScenarioMetadata(UniqueName=name, DynamicsType="HK",
        HKParams=HKParams(Tolerance=tolerance, Influence=influence, RewiringRate=rewiring, RepostRate=repost_rate),
        RecsysFactoryType=recsys_type, RecsysCount=recsys_count, PostRetainCount=post_retain_count,
        MaxSimulationStep=max_sim_step)
