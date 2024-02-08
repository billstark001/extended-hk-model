from base import SimulationParams, HKModelParams
from env import RandomNetworkProvider, ScaleFreeNetworkProvider
from recsys import Random, Opinion, Structure

from dataclasses import asdict

import stats
from w_logger import logger

# model parameters

model_p_random = HKModelParams(
    tolerance=0.4,
    decay=0.1,
    rewiring_rate=0.03,
    recsys_count=10,
    recsys_factory=Random,
)

model_p_opinion = HKModelParams(**asdict(model_p_random))
model_p_opinion.recsys_factory = Opinion

model_p_structure = HKModelParams(**asdict(model_p_random))
model_p_structure.recsys_factory = lambda m: Structure(
    m, matrix_init=True, log=logger.debug)

# simulation parameters

sim_p_standard = SimulationParams(
    max_total_step=6000,
    stat_interval=15,
    opinion_change_error=1e-8,
    stat_collectors={
      'triads': stats.TriadsCountCollector(),
      'cluster': stats.ClusteringCollector(),
      's-index': stats.SegregationIndexCollector(),
      'in-degree': stats.InDegreeCollector(),
      'distance': stats.DistanceCollectorDiscrete(use_js_divergence=True),
    }
)

# network providers

provider_random = RandomNetworkProvider(
    agent_count=4000,
    agent_follow=15,
)

provider_scale_free = ScaleFreeNetworkProvider(
    agent_count=4000,
    agent_follow=15
)

# scenario settings

all_scenarios = {

    'small-scale random network, random rec.sys.': (
        provider_random,
        model_p_random,
        sim_p_standard,
    ),
    'small-scale random network, opinion rec.sys.': (
        provider_random,
        model_p_opinion,
        sim_p_standard,
    ),
    'small-scale random network, structure rec.sys.': (
        provider_random,
        model_p_structure,
        sim_p_standard,
    ),

    'small-scale scale-free network, random rec.sys.': (
        provider_scale_free,
        model_p_random,
        sim_p_standard,
    ),
    'small-scale scale-free network, opinion rec.sys.': (
        provider_scale_free,
        model_p_opinion,
        sim_p_standard,
    ),
    'small-scale scale-free network, structure rec.sys.': (
        provider_scale_free,
        model_p_structure,
        sim_p_standard,
    ),

}
