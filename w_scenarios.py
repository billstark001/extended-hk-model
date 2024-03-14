from base import SimulationParams, HKModelParams
from env import RandomNetworkProvider, ScaleFreeNetworkProvider
from recsys import Random, Opinion, Structure, Mixed

from dataclasses import asdict

import stats
from w_logger import logger

# model parameters

def _p(f):
    return HKModelParams(
        tolerance=0.4,
        decay=0.05,
        rewiring_rate=0.02,
        recsys_count=10,
        recsys_factory=f,
    )


_o = lambda m: Opinion(m)

_s = lambda m: Structure(m, matrix_init=True, log=logger.debug)

_mix = lambda ratio: (lambda m: Mixed(
    m,
    _o(m),
    _s(m),
    ratio
))


model_p_random = _p(Random)
model_p_opinion = _p(_o)
model_p_structure = _p(_s)
model_p_mix3 = _p(_mix(0.3))
model_p_mix7 = _p(_mix(0.7))

# simulation parameters

sim_p_standard = SimulationParams(
    max_total_step=10000,
    stat_interval=15,
    opinion_change_error=1e-5,
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

    'random network, random rec.sys.': (
        provider_random,
        model_p_random,
        sim_p_standard,
    ),
    'random network, opinion rec.sys.': (
        provider_random,
        model_p_opinion,
        sim_p_standard,
    ),
    'random network, structure rec.sys.': (
        provider_random,
        model_p_structure,
        sim_p_standard,
    ),
    'random network, mix3': (
        provider_random,
        model_p_mix3,
        sim_p_standard,
    ),
    'random network, mix7': (
        provider_random,
        model_p_mix7,
        sim_p_standard,
    ),


    'scale-free network, random rec.sys.': (
        provider_scale_free,
        model_p_random,
        sim_p_standard,
    ),
    'scale-free network, opinion rec.sys.': (
        provider_scale_free,
        model_p_opinion,
        sim_p_standard,
    ),
    'scale-free network, structure rec.sys.': (
        provider_scale_free,
        model_p_structure,
        sim_p_standard,
    ),
    'scale-free network, mix3': (
        provider_scale_free,
        model_p_mix3,
        sim_p_standard,
    ),
    'scale-free network, mix7': (
        provider_scale_free,
        model_p_mix7,
        sim_p_standard,
    ),

}
