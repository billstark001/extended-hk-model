from typing import List, Tuple, cast

import os
import pickle

import networkx as nx
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.plot import plot_network_snapshot, plt_figure
import works.detail.simulate as p

from works.stat.context import c

# parameters

scenario_base_path = './run3'
plot_path = './fig3'

os.makedirs(scenario_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

c.set_state(
    active_threshold=0.98,
    min_inactive_value=0.75
)

# utilities

mpl.rcParams['font.size'] = 18


# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"

if __name__ == '__main__':

  # tqdm(p.params_arr, bar_format=short_progress_bar):
  for scenario_name, r, d, g in p.params_arr:

    # load scenario

    scenario_path = os.path.join(
        scenario_base_path, scenario_name + '_record.pkl')
    if not os.path.exists(scenario_path):
      continue

    snapshot_path = os.path.join(plot_path, scenario_name + '_snapshot.png')
    if os.path.exists(snapshot_path):
      continue

    with open(scenario_path, 'rb') as f:
      S_metadata, S_stats, S_agent_stats = pickle.load(f)

    c.set_state(
        model_metadata=S_metadata,
        model_stats=S_stats,
        agent_stats=S_agent_stats,
    )

    S_stat_steps = c.agent_stats_step
    pat_area_hp = c.gradation_index_hp
    steps_smpl = c.steps_smpl
    opinions_smpl = c.opinions_smpl
    graphs_smpl = c.graphs_smpl

    # plot

    fig, all_axes = cast(Tuple[Figure, List[Axes]],
                         plt_figure(n_col=5, total_width=20))
    all_indices = [0, 3, 7, 11, 15]

    plotted_indices = set()
    pos = None
    for i_ax, i in tqdm(enumerate(all_indices), bar_format=short_progress_bar):
      if i in plotted_indices:
        continue
      step = S_stat_steps[i]
      opinion = opinions_smpl[i]
      graph = graphs_smpl[i]
      pos = nx.spring_layout(graph, pos=pos)
      plot_network_snapshot(pos, opinion, graph, all_axes[i_ax], step)
      plotted_indices.add(i)

    title = f'decay: {d:.4f}, rewiring: {r:.4f}, gradation: {pat_area_hp:.6f}'
    plt.title(title)

    plt.savefig(snapshot_path, bbox_inches='tight')
    plt.close()

    print(scenario_name)
