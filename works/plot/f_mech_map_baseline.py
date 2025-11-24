from typing import List
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from result_interp.record import RawSimulationRecord
from utils.plot import plt_figure, plt_save_and_close, setup_paper_params, plot_network_snapshot, plot_opinion_colorbar
import works.config as cfg

# Import reusable functions from f_supp_mech_map
from works.plot.f_supp_mech_map import (
    get_basic_data,
    compute_segmented_potentials,
    eval_rec_x_map,
    eval_rec_dx_map,
    plot_potential_segments,
    set_ax_format,
    plot_colorbar,
    CMAP_NAME,
    PLOT_RES,
)


def plot_baseline_single_row(rec: RawSimulationRecord, save_name: str):
  """Plot baseline scenario in two rows: network snapshots (row 1) and dynamics maps (row 2)"""
  fig, axes = plt_figure(n_row=2, n_col=4, total_width=14)
  axes_r1, axes_r2 = axes
  ax_x, ax_v_x, ax_dx, ax_v_dx = axes_r2

  with rec:
    rec_parsed = get_basic_data(rec)
    
    # Get active_step from cache
    cache = globals().get("basic_data_cache", {})
    active_step = cache.get(rec.unique_name, {}).get("active_step", rec.max_step)
    
    # Calculate snapshot time steps
    snapshot_times_normalized = [0, 0.1, 0.25, 0.5]
    snapshot_times = [int(t * active_step) for t in snapshot_times_normalized]
    
    # Get initial graph and compute spring layout
    G0 = rec.get_graph(0)
    poses = {}
    pos = nx.spring_layout(G0, seed=42)
    poses[0] = pos
    
    # Iteratively refine layout for consistency across snapshots
    for t in snapshot_times[1:]:
      G = rec.get_graph(t)
      pos = nx.spring_layout(G, pos=pos, iterations=10, seed=42)
      poses[t] = pos
    
    # Plot network snapshots
    for i, t in enumerate(snapshot_times):
      G = rec.get_graph(t)
      opinions = rec.opinions[t]
      plot_network_snapshot(
        poses[t], opinions, G, ax=axes_r1[i],
        colorbar=False, node_size=25, alpha=0.2, rasterized=True,
      )
      t_n = snapshot_times_normalized[i]
      axes_r1[i].set_title(f"(a{i+1}) $t_n={t_n}$", loc="left")

  # Plot trajectory-based differential map
  set_ax_format(ax_x, xlabel=True, ylabel=True, dt=1 / PLOT_RES)
  eval_rec_x_map(ax_x, rec_parsed)

  # Plot neighbor-based differential map
  set_ax_format(ax_dx, xlabel=True, ylabel=True, dt="N")
  eval_rec_dx_map(ax_dx, rec_parsed)

  # Compute and plot potentials
  x_grid_x, _, V_segments_x = compute_segmented_potentials(
      rec_parsed, use_neighbor_diff=False, normalize_to_zero=True,
      n_segments=10,
  )
  x_grid_dx, _, V_segments_dx = compute_segmented_potentials(
      rec_parsed, use_neighbor_diff=True, normalize_to_zero=True,
      n_segments=10,
  )

  plot_potential_segments(ax_v_x, x_grid_x, V_segments_x,
                          xlabel=True, ylabel=True)
  plot_potential_segments(
      ax_v_dx, x_grid_dx, V_segments_dx, xlabel=True, ylabel=True)

  # Set titles
  ax_x.set_title("(b) FTOD", loc="left")
  ax_v_x.set_title("(b') Potential of (b)", loc="left")
  ax_dx.set_title("(c) FOD", loc="left")
  ax_v_dx.set_title("(c') Potential of (c)", loc="left")

  # Add colorbars with unified style
  plot_opinion_colorbar(fig, list(axes_r1), label='$x$')
  plot_colorbar(fig, list(axes_r2))

  plt_save_and_close(fig, save_name)


if __name__ == "__main__":
  setup_paper_params()

  # Load baseline scenario (first record)
  rec_baseline = RawSimulationRecord(
      cfg.get_workspace_dir(name="local_mech"), cfg.all_scenarios_mech[0]
  )

  plot_baseline_single_row(rec_baseline, "fig/f_mech_map_baseline")
