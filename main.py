import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import HKModel, HKModelParams
from recsys import Random, Opinion, Structure

n_agent = 1000
n_edges = 20
n_step = 1000

G = nx.erdos_renyi_graph(
    n=n_agent,
    p=n_edges / (n_agent - 1),
    directed=True
)
G_start = nx.DiGraph(G)

params = HKModelParams(
    tolerance=0.3,
    decay=0.1,
    rewiring_rate=0.05,
    recsys_rate=0.8,
    recsys_factory=Opinion,
)

model = HKModel(G, params)
for _ in tqdm(range(n_step)):
  model.step()

data = model.datacollector.get_agent_vars_dataframe()
opinion = data['Opinion'].unstack()

# plt.figure()
# nx.draw_networkx(G)
# plt.show()

# plt.figure()
opinion.plot(lw = 0.5)
plt.legend([])
plt.show()
