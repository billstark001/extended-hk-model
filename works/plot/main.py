import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scipy.stats import gaussian_kde

from works.plot.patterns import (
    load_pattern_data, prepare_heatmap_data, prepare_vectorized_data,
    plot_grad_index_distribution, cat_labels
)
from works.plot.visuals import (
    scatter_heatmap, add_colorbar_legend, heatmap_diff,
    draw_bar_plot, draw_adaptive_moving_stats, plt_figure, plt_save_and_close
)
from works.plot.data_utils import partition_data


def initialize_plotting():
  """Set up plotting parameters"""
  mpl.rcParams['font.size'] = 16
  sns.set_style('whitegrid')


def setup_output_directory(base_path='./fig_final'):
  """Setup output directory for figures"""
  os.makedirs(base_path, exist_ok=True)
  return base_path


def show_fig(name: str, base_path):
  """Save figure to specified path"""
  plt_save_and_close(os.path.join(base_path, name))


def plot_grad_index_heatmap(heatmap_data, base_path):
  """Plot gradation index heatmap"""
  fig, (ax1, ax2, ax3) = plt_figure(n_col=3, hw_ratio=1, total_width=12)

  heatmap_diff(
      fig, ax1, ax2, ax3,
      dict(vmin=0.4, vmax=1, seg=7),
      dict(vmin=-0.3, vmax=0.3, seg=7),
      heatmap_data['grad_index_st'],
      heatmap_data['grad_index_op'],
  )

  show_fig('heatmap_grad_index', base_path)


def plot_grad_index_scatter(vectorized_data, vals_grad_index, base_path):
  """Plot gradation index scatter"""
  fig, ax1 = plt_figure(total_width=5)

  # Split data into opinion and structure
  rd_rate_vec = vectorized_data['rd_rate_vec']
  g_op = vals_grad_index[::2]
  g_st = vals_grad_index[1::2]

  plt.scatter(10 ** rd_rate_vec, g_st, label='structure', s=1.5)
  plt.scatter(10 ** rd_rate_vec, g_op, label='opinion', s=1.5)
  plt.legend()

  h0 = 0.22
  draw_adaptive_moving_stats(
      ax1, rd_rate_vec, g_st, h0=h0, color='tab:blue', log=True)
  draw_adaptive_moving_stats(
      ax1, rd_rate_vec, g_op, h0=h0, color='tab:orange', log=True)

  plt.title('(d) difference', loc='left')
  plt.xscale('log')
  plt.xlabel('rewiring / decay')
  plt.ylabel('gradation index')

  show_fig('scatter_grad_index', base_path)


def plot_active_index_heatmap(heatmap_data, base_path):
  """Plot active index heatmap"""
  fig, (ax1, ax2, ax3) = plt_figure(n_col=3, hw_ratio=1, total_width=12)

  heatmap_diff(
      fig, ax1, ax2, ax3,
      dict(vmin=1.5, vmax=4.5, seg=7),
      dict(vmin=-1, vmax=1, seg=9),
      heatmap_data['active_step_st'],
      heatmap_data['active_step_op'],
  )

  show_fig('heatmap_active_index', base_path)


def plot_triads_heatmap(heatmap_data, base_path):
  """Plot triads heatmap"""
  fig, (ax1, ax2, ax3) = plt_figure(n_col=3, hw_ratio=1, total_width=12)

  heatmap_diff(
      fig, ax1, ax2, ax3,
      dict(vmin=3.5, vmax=4.7, seg=7),
      dict(vmin=-0.6, vmax=0.6, seg=7),
      heatmap_data['triads_st'],
      heatmap_data['triads_op'],
  )

  show_fig('heatmap_triads', base_path)


def plot_event_count_step(vals_0d, vectorized_data, base_path):
  """Plot event count and step"""
  fig, (ax1, ax2) = plt_figure(n_col=2, total_width=12)

  scatter_heatmap(
      ax1,
      np.array(vals_0d['grad_index']),
      np.array(vals_0d['event_count'], dtype=float),
      vectorized_data['is_consensus'],
      vectorized_data['is_not_consensus'],
      vectorized_data['is_near_diag'],
      vectorized_data['is_not_near_diag']
  )

  y = np.array(vals_0d['event_step_mean'] / vals_0d['active_step'])
  y[y > 0.6] = 0.6
  scatter_heatmap(
      ax2,
      np.array(vals_0d['grad_index']), y,
      vectorized_data['is_consensus'],
      vectorized_data['is_not_consensus'],
      vectorized_data['is_near_diag'],
      vectorized_data['is_not_near_diag'],
      legend=True
  )
  add_colorbar_legend(fig)

  for _ in (ax1, ax2):
    _.set_xlabel('gradation index')
  ax1.set_ylabel('#follow event')
  ax2.set_ylabel('time')
  ax1.set_title('(a) total follow event', loc='left')
  ax2.set_title('(b) avg. normalized event time', loc='left')

  show_fig('scatter_event_count_step', base_path)


def plot_bc_hom_g_index(vals_0d, vectorized_data, base_path):
  """Plot BC hom and g index"""
  fig, (ax1, ax2) = plt_figure(n_col=2, total_width=12)
  scatter_heatmap(
      ax1,
      np.array(vals_0d['grad_index']),
      np.array(vals_0d['bc_hom_smpl']),
      vectorized_data['is_consensus'],
      vectorized_data['is_not_consensus'],
      vectorized_data['is_near_diag'],
      vectorized_data['is_not_near_diag']
  )
  scatter_heatmap(
      ax2,
      np.array(vals_0d['grad_index']),
      np.array(vals_0d['g_index_mean_active']),
      vectorized_data['is_consensus'],
      vectorized_data['is_not_consensus'],
      vectorized_data['is_near_diag'],
      vectorized_data['is_not_near_diag'],
      legend=True,
  )
  add_colorbar_legend(fig)

  for _ in (ax1, ax2):
    _.set_xlabel('gradation index')
  ax1.set_ylabel('BC_{hom}')
  ax2.set_ylabel('distance')
  ax1.set_title('(a) bimodality coefficient', loc='left')
  ax2.set_title('(b) avg. subj. opinion distance', loc='left')

  show_fig('scatter_bc_hom_g_index', base_path)


def plot_closed_triads(vals_0d, vectorized_data, base_path):
  """Plot closed triads"""
  triads = np.array(vals_0d['triads'])
  vals_grad_index = np.array(vals_0d['grad_index'])

  def _part(arr: np.ndarray):
    a_op = arr[::2]
    a_st = arr[1::2]
    return a_op, a_st

  tr_op, tr_st = _part(triads)
  g_op, g_st = _part(vals_grad_index)
  c_op, c_st = _part(vectorized_data['is_consensus'])
  nc_op, nc_st = _part(vectorized_data['is_not_consensus'])
  d_op, d_st = _part(vectorized_data['is_near_diag'])
  nd_op, nd_st = _part(vectorized_data['is_not_near_diag'])

  fig, (axfreq, axst2, axop2) = plt_figure(
      n_col=3, hw_ratio=4/5, total_width=18)

  kde_cl_op_ = gaussian_kde(tr_op)
  kde_cl_st_ = gaussian_kde(tr_st)

  metrics = np.linspace(0, 50000, 100)
  kde_cl_op = kde_cl_op_(metrics)
  kde_cl_st = kde_cl_st_(metrics)

  axfreq.plot(metrics, kde_cl_st, label='structure')
  axfreq.plot(metrics, kde_cl_op, label='opinion')
  axfreq.legend()

  scatter_heatmap(axst2, g_st, tr_st, c_st, nc_st, d_st, nd_st)
  scatter_heatmap(axop2, g_op, tr_op, c_op, nc_op, d_op, nd_op, legend=True)

  axfreq.set_title('(a) PDF of #C. T.', loc='left')
  axst2.set_title('(b) structure', loc='left')
  axop2.set_title('(c) opinion', loc='left')

  axfreq.set_xlabel('#closed triads')
  axfreq.set_ylabel('probability')

  for _ in (axst2, axop2):
    _.set_ylabel('#closed triads')
    _.set_xlabel('gradation index')

  fig.tight_layout()
  add_colorbar_legend(fig)
  show_fig('scatter_closed_triads', base_path)


def plot_mean_vars(vals_0d, vectorized_data, base_path):
  """Plot mean variance"""
  fig, (ax_var, ax_scat) = plt_figure(n_col=2, total_width=12)

  y = np.array(vals_0d['mean_vars_smpl'])
  scatter_heatmap(
      ax_scat,
      vals_0d['grad_index'],
      y,
      vectorized_data['is_consensus'],
      vectorized_data['is_not_consensus'],
      vectorized_data['is_near_diag'],
      vectorized_data['is_not_near_diag'],
      legend=True,
  )
  add_colorbar_legend(fig)

  d_means, d_std = partition_data(
      vals_0d['grad_index'], y, vectorized_data['is_consensus'])
  draw_bar_plot(ax_var, cat_labels, d_means, d_std)

  ax_var.set_title('(a) mean variance by categories')
  ax_var.set_ylabel('variance')
  ax_scat.set_title('(b) by gradation index')
  ax_scat.set_xlabel('gradation index')

  show_fig('scatter_mean_vars', base_path)


def main(plot_path='./fig2', base_path='./fig_final'):
  """Main entry point for plotting"""
  # Initialize
  initialize_plotting()
  base_path = setup_output_directory(base_path)

  # Load data
  pat_file_paths = [
      f'{plot_path}/pattern_stats.json',
      f'{plot_path}/pattern_stats_1.json',
      f'{plot_path}/pattern_stats_2.json',
      f'{plot_path}/pattern_stats_3.json',
      f'{plot_path}/pattern_stats_4.json',
  ]

  vals_0d, vals_non_0d = load_pattern_data(pat_file_paths)

  # Prepare data for plots
  heatmap_data = prepare_heatmap_data(vals_0d)
  vectorized_data = prepare_vectorized_data(
      vals_0d, heatmap_data['consensus_threshold'])

  # Generate plots
  plot_grad_index_heatmap(heatmap_data, base_path)
  plot_grad_index_scatter(vectorized_data, vals_0d['grad_index'], base_path)
  plot_active_index_heatmap(heatmap_data, base_path)
  plot_triads_heatmap(heatmap_data, base_path)

  plot_grad_index_distribution(
      vals_0d['grad_index'],
      vectorized_data['is_consensus'],
      vectorized_data['is_not_consensus'],
      os.path.join(base_path, 'dist_grad_consensus')
  )

  plot_event_count_step(vals_0d, vectorized_data, base_path)
  plot_bc_hom_g_index(vals_0d, vectorized_data, base_path)
  plot_closed_triads(vals_0d, vectorized_data, base_path)
  plot_mean_vars(vals_0d, vectorized_data, base_path)


if __name__ == "__main__":
  main()
