from base import SimulationParams, HKModelParams
from env import RandomNetworkProvider
from recsys import Random, Opinion, Structure

from dataclasses import asdict

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
model_p_structure.recsys_factory = Structure

# simulation parameters

sim_p_standard = SimulationParams(
    total_step=800,
    stat_interval=15,
)

# network providers

provider_random = RandomNetworkProvider(
    agent_count=1000,
    agent_follow=15,
)

# scenario settings

all_scenarios = dict(
    random_standard_random=(provider_random, model_p_random, sim_p_standard),
    random_standard_opinion=(provider_random, model_p_random, sim_p_standard),
    random_standard_structure=(
        provider_random, model_p_random, sim_p_standard),
)
