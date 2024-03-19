from typing import Callable, Mapping, cast, Optional, List, Tuple
from numpy.typing import NDArray

import os
from matplotlib.axes import Axes
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

from base import HKModelParams, Scenario, SimulationParams
from base.model import HKModel, HKModelRecommendationSystem
from env import RandomNetworkProvider, ScaleFreeNetworkProvider
from recsys import Random, Opinion, Structure
import stats

from w_logger import logger

BASE_PATH = './fig2'
os.makedirs(BASE_PATH, exist_ok=True)


def plot_and_save_network(
    pos: Mapping,
    opinion: NDArray,
    G: nx.Graph,
    path: str,
    ax_in: Optional[Axes] = None,
    step: int = 0,
    prefix: Optional[str] = None,
    cmap: str = 'coolwarm'
):
  norm = mpl.colors.Normalize(vmin=-1, vmax=1)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])

  mpl.rcParams['font.size'] = 18

  ax = ax_in
  if ax is None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax = cast(Axes, ax)
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['right'].set_visible(False)

  nx.draw_networkx_nodes(G, ax=ax, pos=pos, node_color=opinion,
                         cmap=cmap, vmin=-1, vmax=1, node_size=40)
  nx.draw_networkx_edges(G, ax=ax, pos=pos, node_size=40, alpha=0.36)

  ax.set_xlabel(f't = {step}')

  plt.colorbar(sm, ticks=np.linspace(-1, 1, 5), ax=ax)
  plt.tight_layout()

  if ax_in is None:
    prefix = prefix or 'step'
    fname = '_'.join([str(x) for x in [prefix, step] if x is not None])

    plt.savefig(
        os.path.join(path, fname + '.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


# build scenario


stat_collectors_f = lambda: {
    # 'distance': stats.DistanceCollectorDiscrete(use_js_divergence=True),
    'layout': stats.NetworkLayoutCollector(use_last=True)
}

network_provider = RandomNetworkProvider(
    agent_count=400,
    agent_follow=15,
)
sim_p_standard = SimulationParams(
    max_total_step=1000,
    stat_interval=114514,
    opinion_change_error=1e-4,
    stat_collectors=stat_collectors_f()
)

save_interval = 50

rewiring_rate_array = 10 ** np.arange(-2, 0 + 1/5, 1/4)
decay_rate_array = 10 ** np.arange(-3, 0, 1/3)

n_gens = [
  lambda m: Random(m),
  lambda m: Opinion(m),
  lambda m: Structure(m, sigma=0.3, matrix_init=True)
]

params_arr: List[Tuple[str, float, float, Callable[[HKModel], HKModelRecommendationSystem]]] = []
for i, r in enumerate(rewiring_rate_array):
  for j, d in enumerate(decay_rate_array):
    for k, g in enumerate(n_gens):
      x = (
        f'i{len(params_arr)}_r{i}_d{j}_g{k}_step',
        r,
        d,
        g,
      )
      params_arr.append(x)
    
for name, r, d, g in params_arr[2:]:
  params = HKModelParams(
    tolerance=0.45,
    decay=d,
    rewiring_rate=r,
    recsys_count=10,
    recsys_factory=g
  )
  
  sim_p_standard.stat_collectors = stat_collectors_f()
  S = Scenario(network_provider, params, sim_p_standard)
  S.init()
  
  logger.info('Scenario %s', name)
  
  while not S.check_halt_cond()[0]:
    S.step(save_interval)
    pos, opinion, G = S.collect_stats()['layout']
    plot_and_save_network(pos, opinion, G, BASE_PATH, step=S.steps, prefix=name)
    
    logger.info('Scenario %s, Step %d', name, S.steps)

