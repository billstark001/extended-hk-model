from typing import cast, List, Iterable, Union
from numpy.typing import NDArray

import os
import importlib

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns

from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from utils.file import read_records
from utils.stat import adaptive_moving_stats
from utils.plot import numpy_to_latex_table, plt_figure, get_colormap

import works.gradation.simulate as p
import works.gradation.stat as pp

import utils.plot as _p
importlib.reload(_p)
importlib.reload(p)
importlib.reload(pp)

# figure templates


def linear_func(x, a, b):
  return a * x + b


def plot_line(
    ax: Axes,
    x_data: NDArray, y_data: NDArray
):
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

bool_cmap = LinearSegmentedColormap.from_list("bool", ["tab:red", "tab:blue"])

def scatter_heatmap(
    ax: Axes,
    x: NDArray, y: NDArray,
    c: NDArray, nc: NDArray,
    d: NDArray = None, nd: NDArray = None,
    s=4, lw=.8, h0=.1,
    res=100, res_contour=20,
    xmin=0.4, xmax=1,
    ymin=None, ymax=None,
    legend=False,
):
  if ymin is None:
    ymin = y.min()
  if ymax is None:
    ymax = y.max()
  # only consider c & nc
  xi, yi = np.mgrid[ymin:ymax:res*1j, ymin:ymax:res*1j]
  positions = np.vstack([xi.ravel(), yi.ravel()])
  d_c = gaussian_kde([x[c], y[c]])(positions).reshape(xi.shape)
  d_nc = gaussian_kde([x[nc], y[nc]])(positions).reshape(xi.shape)
  
  s1 = np.sum(c)
  s2 = np.sum(nc)
  d_gross = (d_c * s1 + d_nc * s2) / (s1 + s2)
  d_ratio = d_c / (d_c + d_nc) # 0: nc(polarized), 1: c(consented)
  
  # create heatmap
  d_draw = bool_cmap(d_ratio.T)
  d_draw[..., 3] = d_gross.T / d_gross.max()
  
  # draw density plot
  density_plot = ax.imshow(
    d_draw, interpolation='bilinear',
    aspect='auto',
    extent=[xmin, xmax, ymin, ymax,], 
    origin='lower', 
  )
  
  print([xmin, xmax, ymin, ymax,])
  # contour = ax.contour(
  #   xi, yi, d_gross, 
  #   levels=np.linspace(d_gross.min(), d_gross.max(), res_contour), 
  #   colors='k', linewidths=0.5, alpha=0.7
  # )
  
  # TODO legend
  
  return d_c, d_nc
  

def heatmap_diff(
    fig: Figure,
    ax1: Axes, ax2: Axes, ax3: Axes,
    ax12_params: dict,
    ax3_params: dict,
    heatmap_st: NDArray,
    heatmap_op: NDArray,
):
  cmap_arr5, cmap_setter5 = get_colormap(
      [ax3], cmap='RdBu', fig=fig, **ax3_params, anchor='W')
  cmap_arr4, cmap_setter4 = get_colormap(
      [ax1, ax2], cmap='YlGnBu', **ax12_params, fig=fig, anchor='W')

  ax1.imshow(heatmap_st, **cmap_arr4)
  ax2.imshow(heatmap_op, **cmap_arr4)
  ax3.imshow(heatmap_op - heatmap_st, **cmap_arr5)

  fig.tight_layout()

  for _ in (ax1, ax2, ax3):
    _.invert_yaxis()
    _.set_xticks(np.arange(p.decay_rate_array.size))
    _.set_xticklabels(p.decay_rate_array, rotation=90)
    _.set_yticks(np.arange(p.rewiring_rate_array.size))
    _.set_yticklabels([' ' for _ in p.rewiring_rate_array])
    _.grid(False)

    _.set_xlabel('decay')

  ax1.set_yticklabels(p.rewiring_rate_array)
  ax3.set_yticklabels(p.rewiring_rate_array)

  ax1.set_title('(a) structure', loc='left')
  ax2.set_title('(b) opinion', loc='left')
  ax3.set_title('(c) difference', loc='left')

  ax1.set_ylabel('rewiring')
  # ax3.set_ylabel('rewiring')

  cmap_setter4()
  cmap_setter5()

import matplotlib.pyplot as plt
import numpy as np


def draw_bar_plot(
  ax: Axes,
  labels: List,
  means: NDArray,
  std_devs: NDArray,
  bar_width = 0.5,
  capsize = 5,
):

  bars = ax.bar(labels, means, bar_width, yerr=std_devs, capsize=capsize)

  for bar in bars:
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2., height,
              f'{height:.3f}', ha='center', va='bottom')
  
  return bars

cat_labels = [
  'P;P', 'P;C', 'H;P', 'H;C'
]

def partition_data(
  grad: NDArray, y: NDArray, consented: NDArray, 
  threshold = 0.8,
):
  p2_mask = grad < threshold
  p1_mask = grad >= threshold
  
  polarized = np.logical_not(consented)
  
  d_p2_p = y[np.logical_and(p2_mask, polarized)]
  d_p2_c = y[np.logical_and(p2_mask, consented)]
  d_p1_p = y[np.logical_and(p1_mask, polarized)]
  d_p1_c = y[np.logical_and(p1_mask, consented)]
  
  d = [d_p2_p, d_p2_c, d_p1_p, d_p1_c]
  means = np.array([np.mean(dd) for dd in d])
  std_devs = np.array([np.std(dd) for dd in d])
  return means, std_devs
  

### load data

# parameters
plot_path = './fig2'
pat_file_paths = [
    f'{plot_path}/pattern_stats.json',
    f'{plot_path}/pattern_stats_1.json',
    f'{plot_path}/pattern_stats_2.json',
    f'{plot_path}/pattern_stats_3.json',
    f'{plot_path}/pattern_stats_4.json',
]

mpl.rcParams['font.size'] = 16
sns.set_style('whitegrid')


BASE_PATH = './fig_final'

os.makedirs(BASE_PATH, exist_ok=True)


def plt_save_and_close(path: str):
  plt.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
  plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
  return plt.close()


def show_fig(name: str = 'test'):
  plt_save_and_close(os.path.join(BASE_PATH, name))
  # plt.show()
  # plt.close()

# prepare data


full_sim_len = p.rewiring_rate_array.shape[0] * \
    p.decay_rate_array.shape[0] * len(p.n_gens)

vals_0d, vals_non_0d = read_records(pat_file_paths, full_sim_len)


# plot gradation index graph

# y = vals_0d['event_step_mean'] / vals_0d['active_step']
# y[y > 0.6] = 0.6
# y = np.mean(vals_non_0d['mean_vars_smpl'], axis=1)
# plt.scatter((vals_grad_index ** 3)[::2], y[::2], s=1, label='op')
# plt.scatter((vals_grad_index ** 3)[1::2], y[1::2], s=1, label='st')

vals_0d['mean_vars_smpl'] = np.mean(vals_non_0d['mean_vars_smpl'], axis=1)
vals_0d['bc_hom_smpl'] = np.mean(vals_non_0d['bc_hom_smpl'], axis=1)

vals_grad_index = np.array(vals_0d['grad_index'])
grad_index_median = np.median(vals_grad_index)

# values for partition


# values for heatmap

fields_to_draw_heatmap = ['active_step',
                          'grad_index', 'p_last', 'triads', 'mean_vars_smpl']
pat_csv_values_ = [vals_0d[k].to_numpy(
    dtype=float) for k in fields_to_draw_heatmap]
pat_csv_values_raw = np.array(pat_csv_values_)

pat_csv_values = pat_csv_values_raw.reshape((
    len(fields_to_draw_heatmap),
    -1,
    p.rewiring_rate_array.shape[0],
    p.decay_rate_array.shape[0],
    len(p.n_gens),
))
# axes: (#sim, rewiring, decay, recsys)
active_steps, grad_indices, hs_last, triads, mean_vars = pat_csv_values

# in the following operations, average data is calculated through all simulations

m_active_step = np.mean(active_steps, axis=0, dtype=float)
m_grad_index = np.mean(grad_indices, axis=0, dtype=float)

consensus_threshold = 0.6
m_hs_last = np.mean(hs_last, axis=0, dtype=float)
m_is_consensus = np.mean(hs_last.astype(
    float) < consensus_threshold, axis=0, dtype=float)

m_active_step_op = np.log10(m_active_step[..., 0])
m_active_step_st = np.log10(m_active_step[..., 1])

m_grad_index_op = m_grad_index[..., 0]
m_grad_index_st = m_grad_index[..., 1]


# heat maps

# heatmap of gradation index

fig, (ax1, ax2, ax3) = plt_figure(n_col=3, hw_ratio=1, total_width=12)

heatmap_diff(
    fig, ax1, ax2, ax3,
    dict(vmin=0.4, vmax=1, seg=7),
    dict(vmin=-0.3, vmax=0.3, seg=7),
    m_grad_index_st,
    m_grad_index_op,
)

show_fig('heatmap_grad_index')

# vectorized form of gradation index
# and its scatter plot

rewiring_mat = np.repeat(p.rewiring_rate_array.reshape(
    (-1, 1)), axis=1, repeats=p.decay_rate_array.size)
decay_mat = np.repeat(p.decay_rate_array.reshape(
    (1, -1)), axis=0, repeats=p.rewiring_rate_array.size)
rd_rate_mat = np.log10(rewiring_mat) - np.log10(decay_mat)

# stacked_rd_rate_mat = np.concatenate([
#   rd_rate_mat.reshape(
#     1, p.rewiring_rate_array.shape[0],
#     p.decay_rate_array.shape[0], 1
#   ),
# ] * grad_indices.shape[0], axis=0)

# rd_rate_vec_all, gi_vec_op_all, gi_vec_st_all = np.concatenate([
#   stacked_rd_rate_mat,
#   grad_indices
# ], axis=-1).reshape((-1, 3)).T

rd_rate_vec, gi_vec_st, gi_vec_op = np.array(
    [rd_rate_mat, m_grad_index_st, m_grad_index_op]).reshape((3, -1))

fig, ax1 = plt_figure(total_width=5)
plt.scatter(10 ** rd_rate_vec, gi_vec_st, label='structure', s=1.5)
plt.scatter(10 ** rd_rate_vec, gi_vec_op, label='opinion', s=1.5)
plt.legend()

h0 = 0.22
draw_adaptive_moving_stats(
    ax1, rd_rate_vec, gi_vec_st, h0=h0, color='tab:blue', log=True)
draw_adaptive_moving_stats(
    ax1, rd_rate_vec, gi_vec_op, h0=h0, color='tab:orange', log=True)

plt.title('(d) difference', loc='left')
plt.xscale('log')
plt.xlabel('rewiring / decay')
plt.ylabel('gradation index')

show_fig('scatter_grad_index')

# heatmap of active index

fig, (ax1, ax2, ax3) = plt_figure(n_col=3, hw_ratio=1, total_width=12)

heatmap_diff(
    fig, ax1, ax2, ax3,
    dict(vmin=1.5, vmax=4.5, seg=7),
    dict(vmin=-1, vmax=1, seg=9),
    m_active_step_st,
    m_active_step_op,
)

show_fig('heatmap_active_index')

# heatmap of triads
# deprecated: meaningless

mean_triads = np.mean(triads, axis=0)
mean_triads_op = np.log10(mean_triads[..., 0])
mean_triads_st = np.log10(mean_triads[..., 1])

fig, (ax1, ax2, ax3) = plt_figure(n_col=3, hw_ratio=1, total_width=12)

heatmap_diff(
    fig, ax1, ax2, ax3,
    dict(vmin=3.5, vmax=4.7, seg=7),
    dict(vmin=-0.6, vmax=0.6, seg=7),
    mean_triads_st,
    mean_triads_op,
)

show_fig('heatmap_triads')

# data for partition

# is consensus

is_consensus = np.array(vals_0d['p_last'] < consensus_threshold)
is_not_consensus = np.logical_not(is_consensus)

# is_near_diag
rd_rate_vec_2 = np.vstack([[rd_rate_vec], [rd_rate_vec]]).T.flatten()
rd_rate_vec_all = np.array(
    [rd_rate_vec_2] * int(len(is_consensus) / rd_rate_vec_2.size)).flatten()
is_near_diag = np.logical_and(rd_rate_vec_all > -1, rd_rate_vec_all < 1)
is_not_near_diag = np.logical_not(is_near_diag)

# distributions


# consensus

d = 0.005
metrics = np.arange(0.4, 1 + d, d)
gradation_all = vals_grad_index

kde_nc_raw = gaussian_kde(gradation_all[is_not_consensus])(metrics)
kde_c_raw = gaussian_kde(gradation_all[is_consensus])(metrics)

kde_all_raw = kde_nc_raw + kde_c_raw
kde_all = kde_all_raw / (np.sum(kde_all_raw) * d)
kde_nc = kde_nc_raw / (np.sum(kde_all_raw) * d)
kde_c = kde_all - kde_nc
kde_ratio_c = kde_c / kde_all


fig, (ax_grad, ax_ratio) = plt_figure(n_col=2, total_width=11)

ax_grad.plot(metrics, kde_nc, label='polarized', color='tab:red')
ax_grad.plot(metrics, kde_c, label='consented', color='tab:green')
ax_grad.plot(metrics, kde_all, label='all', color='tab:blue')
ax_grad.legend()

ax_ratio.plot(metrics, kde_ratio_c)

ax_grad.set_title('(a) dist. of gradation index', loc='left')
ax_ratio.set_title('(c) %consented cases', loc='left')

ax_grad.set_yticks(np.array([0, 1, 2, 4, 7, 12]))
for _ in (ax_grad, ax_ratio):
  _.set_xlabel('gradation index')

ax_grad.set_ylabel('prob. density')
ax_ratio.set_ylabel('ratio')

show_fig('dist_grad_consensus')


# scatter plots

# event count
# event step

fig, (ax1, ax2) = plt_figure(n_col=2, total_width=12)
scatter_heatmap(
    ax1,
    np.array(vals_grad_index),
    np.array(vals_0d['event_count'], dtype=float) / 1000,
    is_consensus, is_not_consensus, # is_near_diag, is_not_near_diag
)

y = np.array(vals_0d['event_step_mean'] / vals_0d['active_step'])
y[y > 0.6] = 0.6
scatter_heatmap(
    ax2,
    np.array(vals_grad_index), y,
    is_consensus, is_not_consensus, # is_near_diag, is_not_near_diag,
    legend=True
)

for _ in (ax1, ax2):
  _.set_xlabel('gradation index')
ax1.set_ylabel('#follow event')
ax2.set_ylabel('time')
ax1.set_title('(a) total follow event', loc='left')
ax2.set_title('(b) avg. normalized event time', loc='left')

plt.show()
assert False

show_fig('scatter_event_count_step')


# bc_hom
# subj. dist.

fig, (ax1, ax2) = plt_figure(n_col=2, total_width=12)
scatter_data(
    ax1,
    np.array(vals_grad_index),
    np.array(vals_0d['bc_hom_smpl']),
    is_consensus, is_not_consensus, is_near_diag, is_not_near_diag
)

y = np.array(vals_0d['event_step_mean'] / vals_0d['active_step'])
y[y > 0.6] = 0.6
scatter_data(
    ax2,
    np.array(vals_grad_index),
    np.array(vals_0d['g_index_mean_active']),
    is_consensus, is_not_consensus, is_near_diag, is_not_near_diag,
    legend=True
)

for _ in (ax1, ax2):
  _.set_xlabel('gradation index')
ax1.set_ylabel('BC_{hom}')
ax2.set_ylabel('time')
ax1.set_title('(a) bimodality coefficient', loc='left')
ax2.set_title('(b) avg. subj. opinion distance', loc='left')

show_fig('scatter_bc_hom_g_index')


# count of closed triads

triads = np.array(vals_0d['triads'])


def _part(arr: NDArray):
  a_op = arr[::2]
  a_st = arr[1::2]
  return a_op, a_st


tr_op, tr_st = _part(triads)
g_op, g_st = _part(vals_grad_index)
c_op, c_st = _part(is_consensus)
nc_op, nc_st = _part(is_not_consensus)
d_op, d_st = _part(is_near_diag)
nd_op, nd_st = _part(is_not_near_diag)

fig, (axfreq, axst2, axop2) = plt_figure(n_col=3, hw_ratio=4/5, total_width=18)

kde_cl_op_ = gaussian_kde(tr_op)
kde_cl_st_ = gaussian_kde(tr_st)

metrics = np.linspace(0, 50000, 100)
kde_cl_op = kde_cl_op_(metrics)
kde_cl_st = kde_cl_st_(metrics)

axfreq.plot(metrics, kde_cl_st, label='structure')
axfreq.plot(metrics, kde_cl_op, label='opinion')
axfreq.legend()

tr_st_stat = scatter_data(axst2, g_st, tr_st, c_st, nc_st, d_st, nd_st)
tr_op_stat = scatter_data(axop2, g_op, tr_op, c_op,
                          nc_op, d_op, nd_op, legend=True)

tr_stat_diff = tr_op_stat - tr_st_stat

axfreq.set_title('(a) PDF of #C. T.', loc='left')
axst2.set_title('(b) structure', loc='left')
axop2.set_title('(c) opinion', loc='left')

axfreq.set_xlabel('#closed triads')
axfreq.set_ylabel('probability')

for _ in (axst2, axop2):
  _.set_ylabel('#closed triads')
  _.set_xlabel('gradation index')
  # _.set_ylim(2000, 40000)

plt.tight_layout()
show_fig('scatter_closed_triads')


# mean variance

fig, (ax_var, ax_scat) = plt_figure(n_col=2, total_width=12)

y = np.array(vals_0d['mean_vars_smpl'])
scatter_data(
  ax_scat, vals_grad_index, y,
  is_consensus, is_not_consensus, is_near_diag, is_not_near_diag,
  legend=True,
)

d_means, d_std = partition_data(vals_grad_index, y, is_consensus)

draw_bar_plot(ax_var, cat_labels, d_means, d_std)


ax_var.set_title('(a) mean variance by categories')
ax_var.set_ylabel('variance')
ax_scat.set_title('(b) by gradation index')
ax_scat.set_xlabel('gradation index')

show_fig('scatter_mean_vars')
