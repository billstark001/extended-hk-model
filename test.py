import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from base import HKModelParams, Scenario, SimulationParams
from env import RandomNetworkProvider, ScaleFreeNetworkProvider
from recsys import Random, Opinion, Structure, OpinionRandom
import stats

stat_collectors = {
  'triads': stats.TriadsCountCollector(),
  'cluster': stats.ClusteringCollector(),
  's-index': stats.SegregationIndexCollector(),
  'in-degree': stats.InDegreeCollector(full_data=True),
  'distance': stats.DistanceCollectorDiscrete(use_js_divergence=True),
}

s_params = RandomNetworkProvider(
    agent_count=500,
    agent_follow=15,
)
sim_p_standard = SimulationParams(
    max_total_step=5000,
    model_stat_interval=15,
    model_stat_collectors={}, # stat_collectors,
    agent_stat_keys=['cur_opinion', 'nr_agents', 'op_sum_agents', 'follow_event'],
)

params = HKModelParams(
    tolerance=0.4,
    decay=0.1,
    rewiring_rate=0.03,
    recsys_count=10,
    recsys_factory=lambda m: Random(m), # Structure(m, steepness=None),
)

S = Scenario(s_params, params, sim_p_standard)
S.init()
S.iter()

# plot

metadata, model_stats, agent_stats = S.generate_record_data()
pickle.dumps((metadata, model_stats, agent_stats))

sns.set_theme()

plt.plot(agent_stats['step'], agent_stats['cur_opinion'], lw=0.5)
plt.title('Opinion')
plt.show()

print()