from typing import Callable, Union

from ehk_model_old.base import SimulationParams, HKModelParams
from ehk_model_old.env import RandomNetworkProvider, ScaleFreeNetworkProvider
from ehk_model_old.recsys import Random, Opinion, Structure, Mixed

from dataclasses import asdict

import ehk_model_old.stats as stats

# model parameters

logger: Union[Callable[[str], None], None] = None

def set_logger(l: Union[Callable[[str], None], None]):
    global logger
    logger = l
    
def get_logger():
    global logger
    return logger

dec_rew_pairs = [
    (0.1, 0.01),
    (0.05, 0.02),
    (0.03, 0.03),
    (0.02, 0.05),
    (0.01, 0.1),
]

def _p(f, dec=0.05, rew=0.02):
    return HKModelParams(
        tolerance=0.4,
        decay=dec,
        rewiring_rate=rew,
        recsys_count=10,
        recsys_factory=f,
    )


_o = lambda m: Opinion(m)

_s = lambda m: Structure(m, matrix_init=True, log=get_logger())

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

recsys_o9 = lambda m: Mixed(
        m,
        Random(m, 10),
        Opinion(m),
        0.1)
recsys_s9 = lambda m: Mixed(
        m,
        Random(m, 10),
        Structure(m, noise_std=0.2, matrix_init=False, log=get_logger()),
        0.1)

recsys_o10 = lambda m: Opinion(m)
recsys_s10 = lambda m: Structure(m)

# simulation parameters

sim_p_standard = SimulationParams(
    max_total_step=10000,
    model_stat_interval=15,
    opinion_change_error=1e-5,
    halt_monitor_step=40,
    model_stat_collectors={
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

all_scenarios = {}
for i, p in enumerate(dec_rew_pairs):
    all_scenarios[f'sr{i + 1}'] = (
        provider_random, 
        _p(recsys_s10, *p),
        sim_p_standard
    )
    all_scenarios[f'or{i + 1}'] = (
        provider_random, 
        _p(recsys_o10, *p),
        sim_p_standard
    )
    
for i, p in enumerate(dec_rew_pairs):
    all_scenarios[f'ss{i + 1}'] = (
        provider_scale_free, 
        _p(recsys_s10, *p),
        sim_p_standard
    )
    all_scenarios[f'os{i + 1}'] = (
        provider_scale_free, 
        _p(recsys_o10, *p),
        sim_p_standard
    )
    

all_scenarios_ = {

    'random_random': (
        provider_random,
        model_p_random,
        sim_p_standard,
    ),
    'random_opinion': (
        provider_random,
        model_p_opinion,
        sim_p_standard,
    ),
    'random_structure': (
        provider_random,
        model_p_structure,
        sim_p_standard,
    ),
    'random_mix3': (
        provider_random,
        model_p_mix3,
        sim_p_standard,
    ),
    'random_mix7': (
        provider_random,
        model_p_mix7,
        sim_p_standard,
    ),


    'scale-free_random': (
        provider_scale_free,
        model_p_random,
        sim_p_standard,
    ),
    'scale-free_opinion': (
        provider_scale_free,
        model_p_opinion,
        sim_p_standard,
    ),
    'scale-free_structure': (
        provider_scale_free,
        model_p_structure,
        sim_p_standard,
    ),
    'scale-free_mix3': (
        provider_scale_free,
        model_p_mix3,
        sim_p_standard,
    ),
    'scale-free_mix7': (
        provider_scale_free,
        model_p_mix7,
        sim_p_standard,
    ),

}
