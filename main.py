import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from base import HKModelParams, Scenario, SimulationParams
from env import RandomNetworkProvider, ScaleFreeNetworkProvider
from recsys import Random, Opinion, Structure
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
    recsys_factory=lambda m: Opinion(m),
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

sn = np.std(dn, axis=1)
sr = np.std(dr, axis=1)

plt.plot(steps, sn, lw=1)
plt.plot(steps, sr, lw=1)
plt.plot(steps, sn + sr, lw=1)
plt.legend(['Neighbor', 'Recommended', 'Total'])
plt.title('Standard Deviation of Contribution')
plt.show()

an = np.mean(dn, axis=1)
ar = np.mean(dr, axis=1)

plt.plot(steps, an, lw=1)
plt.plot(steps, ar, lw=1)
plt.plot(steps, an + ar, lw=1)
plt.legend(['Neighbor', 'Recommended', 'Total'])
plt.title('Mean of Contribution')
plt.show()

stats = S.generate_stats()
stats_index = stats['step']

plt.plot(stats_index, stats['triads'])
plt.title('Count of Closed Triads')
plt.show()

plt.plot(stats_index, stats['cluster'])
plt.title('Average Clustering Coefficient')
plt.show()

plt.plot(stats_index, stats['s-index'])
plt.title('Segregation Index')
plt.show()

plt.plot(stats_index, stats['distance-best-o'])
plt.plot(stats_index, stats['distance-best-s'])
plt.plot(stats_index, stats['distance-worst-o'])
plt.plot(stats_index, stats['distance-worst-s'])
plt.legend(['o-best', 's-best', 'o-worst', 's-worst'])
plt.title('JS Divergence of Distance Distribution')
plt.show()

plt.plot(stats_index, stats['in-degree-alpha'])
plt.title('in-degree-alpha')
plt.show()

plt.plot(stats_index, stats['in-degree-p-value'])
plt.title('in-degree-p')
plt.show()

si = stats['in-degree'][-1]
plt.plot(si[0], si[1])
plt.title('in-degree-last')
plt.show()
