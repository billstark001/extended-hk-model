import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from model import HKModel, HKModelParams
from scenario import ScenarioParams, Scenario
from recsys import Random, Opinion, Structure

s_params = ScenarioParams(
    n_agent=1000,
    n_edges=15,
    n_step=300,
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
opinion, dn, dr = S.get_opinion_data()
# (t, n)

# plt.figure()
plt.plot(opinion, lw=0.5)
plt.title('Opinion')
plt.show()

mn = np.mean(np.abs(dn), axis=1)
sn = np.std(dn, axis=1)
mr = np.mean(np.abs(dr), axis=1)
sr = np.std(dr, axis=1)
plt.plot(sn, lw=1)
plt.plot(sr, lw=1)
plt.legend(['Neighbor', 'Recommended'])
plt.title('Variance')
plt.show()
plt.plot(mn, lw=1)
plt.plot(mr, lw=1)
plt.legend(['Neighbor', 'Recommended'])
plt.title('Mean')
plt.show()
