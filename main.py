import networkx as nx

from model import HKModel, HKModelParams

G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

params = HKModelParams()

model = HKModel(G, params)
for _ in range(10):
  model.step()
