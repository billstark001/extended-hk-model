from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from result_interp.record import RawSimulationRecord
from utils.plot import plt_figure, plt_save_and_close, setup_paper_params
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
  """Plot baseline scenario in a single row: x_map, V(x), dx_map, V(dx)"""
  fig, axes = plt_figure(n_row=1, n_col=4, total_width=14)
  ax_x, ax_v_x, ax_dx, ax_v_dx = axes[0], axes[1], axes[2], axes[3]

  with rec:
    rec_parsed = get_basic_data(rec)

  # Plot trajectory-based differential map
  set_ax_format(ax_x, xlabel=True, ylabel=True, dt=1 / PLOT_RES)
  eval_rec_x_map(ax_x, rec_parsed)

  # Plot neighbor-based differential map
  set_ax_format(ax_dx, xlabel=True, ylabel=True, dt="N")
  eval_rec_dx_map(ax_dx, rec_parsed)

  # Compute and plot potentials
  x_grid_x, _, V_segments_x = compute_segmented_potentials(
      rec_parsed, use_neighbor_diff=False, normalize_to_zero=True
  )
  x_grid_dx, _, V_segments_dx = compute_segmented_potentials(
      rec_parsed, use_neighbor_diff=True, normalize_to_zero=True
  )

  plot_potential_segments(ax_v_x, x_grid_x, V_segments_x,
                          xlabel=True, ylabel=True)
  plot_potential_segments(
      ax_v_dx, x_grid_dx, V_segments_dx, xlabel=True, ylabel=True)

  # Set titles
  ax_x.set_title("(b) Trajectory diff.", loc="left")
  ax_v_x.set_title("(b') Potential of (b)", loc="left")
  ax_dx.set_title("(c) Neighbor diff.", loc="left")
  ax_v_dx.set_title("(c') Potential of (c)", loc="left")

  # Add colorbar
  plot_colorbar(fig, list(axes))

  plt_save_and_close(fig, save_name)


if __name__ == "__main__":
  setup_paper_params()

  # Load baseline scenario (first record)
  rec_baseline = RawSimulationRecord(
      cfg.get_workspace_dir(name="local_mech"), cfg.all_scenarios_mech[0]
  )

  plot_baseline_single_row(rec_baseline, "fig/f_mech_map_baseline")
