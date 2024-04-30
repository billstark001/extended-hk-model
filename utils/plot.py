from typing import Mapping, Optional, Tuple, List, overload, Union, Iterable, Any
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

  ax.set_xlabel(f'step = {step}')

  plt.colorbar(sm, ticks=np.linspace(-1, 1, 5), ax=ax)
  plt.tight_layout()


@overload
def plt_figure(n_row: int,
               hw_ratio=3/4, total_width=16) -> Tuple[Figure, List[Axes]]: ...


@overload
def plt_figure(n_col: int,
               hw_ratio=3/4, total_width=16) -> Tuple[Figure, List[Axes]]: ...


@overload
def plt_figure(
    n_row: int, n_col: int,
    hw_ratio=3/4, total_width=16,
    **kwargs
) -> Tuple[Figure, List[List[Axes]]]: ...


def plt_figure(
    n_row=1, n_col=1,
    hw_ratio=3/4, total_width=16,
    **kwargs
) -> Tuple[Figure, List[List[Axes]]]:
  width = total_width / n_col
  height = width * hw_ratio
  total_height = height * n_row
  return plt.subplots(n_row, n_col, figsize=(total_width, total_height), **kwargs)


def get_colormap(
    axes: Union[Axes, Iterable[Axes]],
    cmap='YlGnBu',
    vmin: float = -1, vmax: float = 1, seg: int = 5,
    fig: Any = plt,
    **kwargs
):
  norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])

  cmap_arr = dict(cmap=cmap, vmin=vmin, vmax=vmax)

  def set_cmap_func(): return fig.colorbar(
      sm, ticks=np.linspace(vmin, vmax, seg), ax=axes, **kwargs)

  return cmap_arr, set_cmap_func


def numpy_to_latex_table(
    data: NDArray,
    filename: str,
    caption="Table",
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    scientific=True,
    precision=4
):
  # convert to string
  formatted_data = []
  for row in data:
    formatted_row = []
    for val in row:
      if scientific:
        formatted_row.append(f"{val:.{precision}e}")
      else:
        formatted_row.append(f"{val:.{precision}f}")
    formatted_data.append(formatted_row)

  # create table
  table = "\\begin{table}[h]\n\\centering\n"
  if row_labels is not None and col_labels is not None:
    table += "\\begin{tabular}{|l|" + "c|" * (data.shape[1]-1) + "c|}\n"
    table += "\\hline\n"
    table += " & " + " & ".join(col_labels) + "\\\\\n\\hline\n"
    for i, row in enumerate(formatted_data):
      table += row_labels[i] + " & " + " & ".join(row) + "\\\\\n\\hline\n"
  else:
    table += "\\begin{tabular}{" + "c" * data.shape[1] + "}\n"
    for row in formatted_data:
      table += " & ".join(row) + "\\\\\n"
  table += "\\end{tabular}\n\\caption{" + caption + "}\n\\end{table}"

  # save table
  with open(filename, "w") as f:
    f.write(table)
