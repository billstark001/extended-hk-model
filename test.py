import networkx as nx
import numpy as np
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
    max_total_step=1000,
    stat_interval=15,
    stat_collectors=stat_collectors
)

params = HKModelParams(
    tolerance=0.4,
    decay=0.1,
    rewiring_rate=0.03,
    recsys_count=10,
    recsys_factory=lambda m: Structure(m, steepness=None),
)

S = Scenario(s_params, params, sim_p_standard)
S.init()
S.step()

# plot

sns.set_theme()

steps, opinion, dn, dr, sum_n, sum_r, n_n, n_r = S.get_opinion_data()  # (t, n)

plt.plot(opinion, lw=0.5)
plt.title('Opinion')
plt.show()
