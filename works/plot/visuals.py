from typing import List
from numpy.typing import NDArray

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

from utils.stat import adaptive_moving_stats

from works.plot.utils import linear_func, scale_array


def plot_line(
    ax: Axes,
    x_data: NDArray, y_data: NDArray
):
  """Plot a linear best fit line through data points"""
  from scipy.optimize import curve_fit

  (a, b), _ = curve_fit(linear_func, x_data, y_data)
  min_x = np.min(x_data)
  max_x = np.max(x_data)
  x_line = np.array([min_x, max_x])
  ax.plot(x_line, a * x_line + b)
  return a, b


def draw_adaptive_moving_stats(
    axes: Axes,
    x: NDArray,
    y: NDArray,
    h0=.1,
    color='tab:gray',
    alpha=0.1,
    log=False,
):
  """Draw adaptive moving statistics (mean and std)"""
  _points, mean, var = adaptive_moving_stats(x, y, h0)
  points = 10 ** _points if log else _points
  std = var ** 0.5
  axes.plot(points, mean, lw=1, color=color)
  axes.fill_between(
      points, mean - std, mean + std, color=color, alpha=alpha)
  return points, mean, std


def scatter_data(
    ax: Axes,
    x: NDArray, y: NDArray,
    c: NDArray, nc: NDArray,
    d: NDArray, nd: NDArray,
    s=4, lw=.8, h0=.1,
    legend=False,
):
  """Create a scatter plot with data categorized by c/nc and d/nd"""
  _1 = np.logical_and(nc, nd)
  _2 = np.logical_and(c, nd)
  _3 = np.logical_and(nc, d)
  _4 = np.logical_and(c, d)
  ax.scatter(x[_1], y[_1],
             color='tab:blue', marker='x', label='ND;P', s=s, linewidths=lw)
  ax.scatter(x[_2], y[_2],
             color='tab:cyan', marker='+', label='ND;C', s=s * 1.25, linewidths=lw)
  ax.scatter(x[_3], y[_3],
             color='tab:red', marker='x', label='D;P', s=s, linewidths=lw)
  ax.scatter(x[_4], y[_4],
             color='tab:orange', marker='+', label='D;C', s=s * 1.25, linewidths=lw)
  if legend:
    ax.legend(bbox_to_anchor=(1.05, 1))

  if x[nc].size:
    draw_adaptive_moving_stats(ax, x[nc], y[nc], h0)
  if x[c].size:
    draw_adaptive_moving_stats(ax, x[c], y[c], h0)

  return np.array([[np.mean(z[_i]) for _i in [_1, _2, _3, _4]] for z in [x, y]])


# Define color maps
bool_cmap = LinearSegmentedColormap.from_list(
    "bool", ["tab:red", 'tab:purple', "tab:blue"])
density_cmap = LinearSegmentedColormap.from_list("bool", ["white", "gray"])


def scatter_heatmap(
    ax: Axes,
    x: NDArray, y: NDArray,
    c: NDArray, nc: NDArray,
    d: NDArray = None, nd: NDArray = None,
    s=4, lw=.4, h0=.1,
    res=100, res_contour=15,
    xmin=0.4, xmax=1,
    ymin=None, ymax=None,
    pat_part=0.8,
    legend=False,
    scale=True
):
  """Create a scatter plot with heatmap showing density"""
  if ymin is None:
    ymin = y.min()
  if ymax is None:
    ymax = y.max()

  xi, yi = np.mgrid[xmin:xmax:res*1j, ymin:ymax:res*1j]

  if scale:
    y_scaled = scale_array(y, (ymin, ymax), (xmin, xmax))
    ymin_scaled, ymax_scaled = (xmin, xmax)
    yi_scaled = scale_array(yi, (ymin, ymax), (xmin, xmax))
  else:
    y_scaled = y
    ymin_scaled, ymax_scaled = (ymin, ymax)
    yi_scaled = yi

  # only consider c & nc
  from scipy.stats import gaussian_kde

  positions = np.vstack([xi.ravel(), yi_scaled.ravel()])
  d_c = gaussian_kde(np.array([x[c], y_scaled[c]]))(
      positions).reshape(xi.shape)
  d_nc = gaussian_kde(np.array([x[nc], y_scaled[nc]]))(
      positions).reshape(xi.shape)

  s1 = np.sum(c)
  s2 = np.sum(nc)
  d_gross = (d_c * s1 + d_nc * s2) / (s1 + s2)
  # 0: nc(polarized), 1: c(consensual)
  d_ratio = d_c * (s1 / (s1 + s2)) / d_gross

  # create heatmap
  d_draw = bool_cmap(d_ratio.T)
  d_draw[..., 3] = d_gross.T / d_gross.max()

  # find maxima
  from scipy import ndimage

  max_filtered = ndimage.maximum_filter(d_gross, size=10)
  maxima = (d_gross == max_filtered) & (d_gross > np.mean(d_gross))
  max_coords = np.column_stack(np.where(maxima))

  # draw density plot
  density_plot = ax.imshow(
      d_draw, interpolation='bilinear',
      aspect='auto',
      extent=[xmin, xmax, ymin, ymax,],
      origin='lower',
  )

  contour = ax.contour(
      xi, yi, d_gross,
      levels=np.linspace(d_gross.min(), d_gross.max(), res_contour),
      colors='k', linewidths=lw, alpha=0.7
  )

  # annotate
  for coord in max_coords:
    y, x = coord
    xc, yc = (xi[y, x], yi[y, x])
    ax.plot(xc, yc, '+', markersize=10, color='black')
    ax.annotate(
        f'({xc:.2f}, {yc:.2f})',
        (xc, yc),
        xytext=(-42, -18) if xc < pat_part else (-90, -10),
        textcoords='offset points',
    )

  return d_c, d_nc


def add_colorbar_legend(fig: Figure, density_vmax=1.):
  """Add a color bar legend to the figure"""
  fig.subplots_adjust(right=0.85)
  cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])

  sm = plt.cm.ScalarMappable(
      cmap=bool_cmap, norm=plt.Normalize(vmin=0, vmax=100))
  sm.set_array([])
  cbar_density = fig.colorbar(sm, cax=cbar_ax)
  cbar_density.set_label('% consensual cases')


def heatmap_diff(
    fig: Figure,
    ax1: Axes, ax2: Axes, ax3: Axes,
    ax12_params: dict,
    ax3_params: dict,
    heatmap_st: NDArray,
    heatmap_op: NDArray,
):
  """Create a heatmap showing difference between structure and opinion"""
  from utils.plot import get_colormap

  cmap_arr5, cmap_setter5 = get_colormap(
      [ax3], cmap='RdBu', fig=fig, **ax3_params, anchor='W')
  cmap_arr4, cmap_setter4 = get_colormap(
      [ax1, ax2], cmap='YlGnBu', **ax12_params, fig=fig, anchor='W')

  ax1.imshow(heatmap_st, **cmap_arr4)
  ax2.imshow(heatmap_op, **cmap_arr4)
  ax3.imshow(heatmap_op - heatmap_st, **cmap_arr5)

  fig.tight_layout()

  import works.config as cfg

  for _ in (ax1, ax2, ax3):
    _.invert_yaxis()
    _.set_xticks(np.arange(cfg.decay_rate_array.size))
    _.set_xticklabels(cfg.decay_rate_array, rotation=90)
    _.set_yticks(np.arange(cfg.rewiring_rate_array.size))
    _.set_yticklabels([' ' for _ in cfg.rewiring_rate_array])
    _.grid(False)

    _.set_xlabel('decay')

  ax1.set_yticklabels(cfg.rewiring_rate_array)
  ax3.set_yticklabels(cfg.rewiring_rate_array)

  ax1.set_title('(a) structure', loc='left')
  ax2.set_title('(b) opinion', loc='left')
  ax3.set_title('(c) difference', loc='left')

  ax1.set_ylabel('rewiring')

  cmap_setter4()
  cmap_setter5()


def draw_bar_plot(
    ax: Axes,
    labels: List,
    means: NDArray,
    std_devs: NDArray,
    bar_width=0.5,
    capsize=5,
):
  """Draw a bar plot with error bars"""
  bars = ax.bar(labels, means, bar_width, yerr=std_devs, capsize=capsize)

  for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom')

  return bars


def plt_figure(*args, **kwargs):
  """Wrapper around plt.subplots for consistent figure styling"""
  return plt.subplots(*args, **kwargs)


def plt_save_and_close(path: str):
  """Save figure to path and close it"""
  plt.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
  plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
  return plt.close()
