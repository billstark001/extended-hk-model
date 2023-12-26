import networkx as nx
import matplotlib.pyplot as plt
from model import HKModel, HKModelParams
from scenario import ScenarioParams, Scenario
from recsys import Random, Opinion, Structure

s_params = ScenarioParams(
    n_agent=1000,
    n_edges=15,
    n_step=200,
)

params = HKModelParams(
    tolerance=0.4,
    decay=0.1,
    rewiring_rate=0.03,
    recsys_count=10,
    recsys_factory=Random,
)

S = Scenario(s_params, params)
S.init()
S.step()
opinion = S.get_opinion_data()
# plt.figure()
# nx.draw_networkx(G)
# plt.show()

# plt.figure()
opinion.plot(lw=0.5)
plt.legend([])
plt.show()
