from typing import Mapping, Optional, Tuple, List
from numpy.typing import NDArray

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def plot_network_snapshot(
    pos: Mapping,
    opinion: NDArray,
    G: nx.Graph,
    ax: Optional[Axes] = None,
    step: int = 0,
    cmap: str = 'coolwarm'
):
  norm = mpl.colors.Normalize(vmin=-1, vmax=1)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])

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
  

def plt_figure(
  n_row=1, n_col=1, 
  hw_ratio=3/4, total_width=16, 
  **kwargs
) -> Tuple[Figure, List[Axes]]:
  width = total_width / n_col
  height = width * hw_ratio
  total_height = height * n_row
  return plt.subplots(n_row, n_col, figsize=(total_width, total_height), **kwargs)
