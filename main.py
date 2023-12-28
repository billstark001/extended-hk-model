import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import HKModel, HKModelParams
from scenario import ScenarioParams, Scenario
from recsys import Random, Opinion, Structure

s_params = ScenarioParams(
    agent_count=1000,
    agent_follow=15,
    total_step=500,
    stat_interval=15,
)

params = HKModelParams(
    tolerance=0.4,
    decay=0.1,
    rewiring_rate=0.03,
    recsys_count=10,
    recsys_factory=Structure,
)

S = Scenario(s_params, params)
S.init()
S.step()

# plot

sns.set()

opinion, dn, dr = S.get_opinion_data()
# (t, n)

# plt.figure()
plt.plot(opinion, lw=0.5)
plt.title('Opinion')
plt.show()

mn = np.mean(dn, axis=1)
sn = np.std(dn, axis=1)
mr = np.mean(dr, axis=1)
sr = np.std(dr, axis=1)

plt.plot(sn, lw=1)
plt.plot(sr, lw=1)
plt.plot(sn + sr, lw=1)
plt.legend(['Neighbor', 'Recommended', 'Total'])
plt.title('Standard Deviation')
plt.show()

# plt.plot(mn, lw=1)
# plt.plot(mr, lw=1)
# plt.plot(mn + mr, lw=1)
# plt.legend(['Neighbor', 'Recommended', 'Total'])
# plt.title('Mean')
# plt.show()

stats = S.stats
stats_index = sorted(stats.keys())
distance, triads, clustering, segregation = [
    [stats[i][n] for i in stats_index] \
    for n in range(4)
]

plt.plot(stats_index, triads)
plt.title('Count of Closed Triads')
plt.show()

plt.plot(stats_index, clustering)
plt.title('Average Clustering Coefficient')
plt.show()

plt.plot(stats_index, segregation)
plt.title('Segregation Index')
plt.show()