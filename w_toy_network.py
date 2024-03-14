import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from base import HKModelParams, Scenario, SimulationParams
from env import RandomNetworkProvider, ScaleFreeNetworkProvider
from recsys import Random, Opinion, Structure
import stats

raise Exception('do not run this file again')

stat_collectors = {
    # 'distance': stats.DistanceCollectorDiscrete(use_js_divergence=True),
    'layout': stats.NetworkLayoutCollector()
}

s_params = RandomNetworkProvider(
    agent_count=400,
    agent_follow=15,
)
sim_p_standard = SimulationParams(
    max_total_step=114514,
    stat_interval=114514,
    stat_collectors=stat_collectors
)

params = HKModelParams(
    tolerance=0.4,
    decay=0.04,
    rewiring_rate=0.02,
    recsys_count=10,
    recsys_factory=lambda m: Opinion(m),
)


BASE_PATH = './fig_final'

os.makedirs(BASE_PATH, exist_ok=True)

def plt_save_and_close(path: str):
  plt.savefig(path + '.eps', dpi=300, bbox_inches='tight')
  plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
  return plt.close()

S = Scenario(s_params, params, sim_p_standard)
step = 300
S.init()
l0 = S.collect_stats()['layout']
S.step(step)
l1 = S.collect_stats()['layout']
S.step(step)
l2 = S.collect_stats()['layout']
S.step(step // 2)

x = 12
width = x / 3
height = width * 3 / 4
cmap = 'coolwarm'

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

mpl.rcParams['font.size'] = 18

fig, axes = plt.subplots(1, 3, figsize=(x, height))
for i, (pos, color, G) in enumerate([l0, l1, l2]):
  ax = axes[i]
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['right'].set_visible(False)
  nx.draw_networkx_nodes(
      G, ax=ax, pos=pos, node_color=color, cmap=cmap, vmin=-1, vmax=1, node_size=40)
  nx.draw_networkx_edges(G, ax=ax, pos=pos, node_size=40, alpha=0.36)
  ax.set_xlabel(f't = {i * step}')

plt.colorbar(sm, ticks=np.linspace(-1, 1, 5), ax=axes[2])
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt_save_and_close(os.path.join(BASE_PATH, 'toy_network'))


sns.set_theme()
steps, opinion, dn, dr = S.get_opinion_data()  # (t, n)
plt.figure(figsize=(4, 3))
plt.plot(opinion, lw=0.5)
plt.title('Opinion')

plt_save_and_close(os.path.join(BASE_PATH, 'toy_opinion'))
