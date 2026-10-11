"""Microscopic network snapshots with a shared opinion-color convention."""

from typing import Iterable, List, Mapping, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

PAPER_WIDTH = 12.0  # inches
DEFAULT_HW_RATIO = 4 / 5  # height / width


def plot_network_snapshot(
    pos: Mapping | None,
    opinion: NDArray,
    G: nx.Graph,
    ax: Optional[Axes] = None,
    step: int | float | None = None,
    cmap: str = "coolwarm",
    colorbar: bool | Iterable[Axes] = True,
    node_size: int = 40,
    alpha: float = 0.36,
    rasterized: bool = True,
):
    """Draw G with node colors for opinions on [-1,1]; compute a spring layout when pos is absent and optionally attach a colorbar."""
    if pos is None:
        pos = nx.spring_layout(G, pos=pos)
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)  # type: ignore
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if ax is not None:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

    nx.draw_networkx_nodes(
        G,
        ax=ax,
        pos=pos,
        node_color=opinion,  # type: ignore
        cmap=cmap,
        vmin=-1,
        vmax=1,
        node_size=node_size,
    )
    nx.draw_networkx_edges(G, ax=ax, pos=pos, node_size=node_size, alpha=alpha)
    if ax is not None and rasterized:
        ax.set_rasterized(True)

    if ax is not None and step is not None:
        ax.set_xlabel(f"step = {step}")

    if colorbar:
        ax_colorbar = ax if colorbar is True else colorbar  # type: ignore
        if ax_colorbar is not None:
            plt.colorbar(sm, ticks=np.linspace(-1, 1, 5), ax=ax_colorbar)


def plot_opinion_colorbar(fig: Figure, axes: Union[Axes, List[Axes]], cmap: str = "coolwarm", label: str = "Opinion"):
    """Plot a unified colorbar for opinion values across multiple axes

    Args:
      fig: The figure object
      axes: Single axis or list of axes to attach the colorbar to
      cmap: Colormap name (default: 'coolwarm')
      label: Label for the colorbar (default: 'Opinion')
    """
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)  # type: ignore
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    axes_list = axes if isinstance(axes, list) else [axes]
    cbar = fig.colorbar(sm, ax=axes_list, ticks=np.linspace(-1, 1, 5), orientation="vertical", aspect=30, pad=0.01)
    cbar.set_label(label)
    return cbar
