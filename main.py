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

params = HKModelParams(
  tolerance=0.3,
  decay=0.3,
  rewiring_rate=0.3,
  recsys_factory=lambda _: Random(_)
)

model = HKModel(G, params)
for _ in tqdm(range(100)):
  model.step()

data = model.datacollector.get_agent_vars_dataframe()
opinion = data['Opinion'].unstack()
opinion.plot()
plt.show()
