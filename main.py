import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import HKModel, HKModelParams
from recsys import Random

G = nx.erdos_renyi_graph(
  n=200,
  p=0.2,
  directed=True
)
G_start = nx.DiGraph(G)

params = HKModelParams(
  tolerance=0.3,
  decay=0.1,
  rewiring_rate=0.3,
  recsys_factory=lambda _: Random(_)
)

model = HKModel(G, params)
for _ in tqdm(range(100)):
  model.step()

data = model.datacollector.get_agent_vars_dataframe()
opinion = data['Opinion'].unstack()

# plt.figure()
# nx.draw_networkx(G)
# plt.show()

# plt.figure()
opinion.plot()
plt.show()
