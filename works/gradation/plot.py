from typing import cast, List, Iterable, Union
from numpy.typing import NDArray

import os

import json
import importlib

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
import seaborn as sns

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from base import Scenario

import works.gradation.simulate as p
import works.gradation.stat as pp

import utils.plot as _p
importlib.reload(_p)
importlib.reload(p)
importlib.reload(pp)

from utils.plot import numpy_to_latex_table, plt_figure, get_colormap

# parameters

plot_path = './fig2'
pat_file_paths = [
  './fig2/pattern_stats.json',
  './fig2/pattern_stats_2.json',
]

mpl.rcParams['font.size'] = 16
sns.set_style('whitegrid')


BASE_PATH = './fig_final'

os.makedirs(BASE_PATH, exist_ok=True)


def plt_save_and_close(path: str):
  plt.savefig(path + '.eps', dpi=300, bbox_inches='tight')
  plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
  return plt.close()


def show_fig(name: str = 'test'):
  plt_save_and_close(os.path.join(BASE_PATH, name))
  # plt.show()
  # plt.close()
  
# prepare data

full_sim_len = p.rewiring_rate_array.shape[0] * p.decay_rate_array.shape[0] * len(p.n_gens)

pat_files_raw = []
for f in pat_file_paths:
  if not os.path.exists(f):
    continue
  with open(f, 'r', encoding='utf8') as fp:
    pat_files_ = json.load(fp)
    total_len = len(pat_files_)
    used_len = total_len - total_len % full_sim_len
    pat_files_raw += pat_files_[:used_len]
  

keys = ['name', 'active_step', 'pat_abs_mean', 'pat_area_hp', 'p_last', 'g_index_mean_active']
pat_csv_values_ = [[x[key] for key in keys] for x in pat_files_raw]
pat_csv_values_raw = np.array(pat_csv_values_)
    
pat_csv_values = pat_csv_values_raw.T.reshape((
  len(keys),
  -1,
  p.rewiring_rate_array.shape[0],
  p.decay_rate_array.shape[0],
  len(p.n_gens),
))
# axes: (#sim, rewiring, decay, recsys)
names, active_steps, abs_hp, areas_hp, hs_last, g_index_mean_active = pat_csv_values

# in the following operations, average data is calculated through all simulations

m_active_step = np.mean(active_steps, axis=0, dtype=float)
m_pattern_1_abs = np.mean(abs_hp, axis=0, dtype=float)
m_pattern_1_areas = np.mean(areas_hp, axis=0, dtype=float)

consensus_threshold = 0.6
m_hs_last = np.mean(hs_last, axis=0, dtype=float)
m_is_consensus = np.mean(hs_last.astype(float) < consensus_threshold, axis=0, dtype=float)

# m_pattern_1_op = means[..., 0, :].astype(float)
# m_pattern_1_st = means[..., 1, :].astype(float)
# m_hs_last_op = hs_last[..., 0, :].astype(float)
# m_hs_last_st = hs_last[..., 1, :].astype(float)


m_pattern_1_op = m_pattern_1_abs[..., 0]
m_pattern_1_st = m_pattern_1_abs[..., 1]
m_pattern_1_a_op = m_pattern_1_areas[..., 0]
m_pattern_1_a_st = m_pattern_1_areas[..., 1]
m_hs_last_op = m_hs_last[..., 0]
m_hs_last_st = m_hs_last[..., 1]
m_is_consensus_op = m_is_consensus[..., 0]
m_is_consensus_st = m_is_consensus[..., 1]

rewiring_mat = np.repeat(p.rewiring_rate_array.reshape((-1, 1)), axis=1, repeats=p.decay_rate_array.size)
decay_mat = np.repeat(p.decay_rate_array.reshape((1, -1)), axis=0, repeats=p.rewiring_rate_array.size)

rd_rate_mat = np.log10(rewiring_mat) - np.log10(decay_mat)

rd_rate_vec, pat1_st_vec, pat1_op_vec = np.array([rd_rate_mat, m_pattern_1_a_st, m_pattern_1_a_op]).reshape((3, -1))

# figure 1

fig, (ax1, ax2, ax3) = plt_figure(n_col=3, hw_ratio=1, total_width=12)

cmap_arr5, cmap_setter5 = get_colormap([ax3], cmap='RdBu', fig=fig, vmin=-0.3, vmax=0.3, anchor='W')
cmap_arr4, cmap_setter4 = get_colormap([ax1, ax2], cmap='YlGnBu', vmin=0.3, seg=8, fig=fig, anchor='W')
  
ax1.imshow(m_pattern_1_a_st, **cmap_arr4)
ax2.imshow(m_pattern_1_a_op, **cmap_arr4)
ax3.imshow(m_pattern_1_a_op - m_pattern_1_a_st, **cmap_arr5)

fig.tight_layout()

# ax4.imshow(m_pattern_1_a_op - m_pattern_1_a_st > 0, **cmap_arr5)

for _ in (ax1, ax2, ax3):
  _.invert_yaxis()
  _.set_xticks(np.arange(p.decay_rate_array.size))
  _.set_xticklabels(p.decay_rate_array, rotation=90)
  _.set_yticks(np.arange(p.rewiring_rate_array.size))
  _.set_yticklabels([' ' for _ in p.rewiring_rate_array])
  _.grid(False)
  
  _.set_xlabel('decay')
  
# bbox = ax3.get_position(original=True)
# dx = 0.01
# ax3.set_position([bbox.xmin - dx, bbox.ymin, bbox.xmax - dx - bbox.xmin, bbox.ymax - bbox.ymin], which='original')
  
  
ax1.set_yticklabels(p.rewiring_rate_array)
ax3.set_yticklabels(p.rewiring_rate_array)

ax1.set_title('(a) structure', loc='left')
ax2.set_title('(b) opinion', loc='left')
ax3.set_title('(c) difference', loc='left')

ax1.set_ylabel('rewiring')
# ax3.set_ylabel('rewiring')

cmap_setter4()
cmap_setter5()

show_fig('grad_stat_heatmap')


fig, ax1 = plt_figure(total_width=5)
plt.scatter(10 ** rd_rate_vec, pat1_st_vec, label='structure', s=1.5)
plt.scatter(10 ** rd_rate_vec, pat1_op_vec, label='opinion', s=1.5)
plt.legend()
plt.title('(d) difference', loc='left')
plt.xscale('log')
plt.xlabel('rewiring / decay')
plt.ylabel('gradation index')

show_fig('grad_stat_scatter')


# figure 2

def linear_func(x, a, b):
  return a * x + b

def plot_line(x_data, y_data):
  (a, b), _ = curve_fit(linear_func, x_data, y_data)
  min_x = np.min(x_data)
  max_x = np.max(x_data)
  x_line = np.array([min_x, max_x])
  plt.plot(x_line, a * x_line + b)
  return a, b

# x1, y1 = m_pattern_1_a_op.flatten(), m_is_consensus_op.flatten()
# x2, y2 = m_pattern_1_a_st.flatten(), m_is_consensus_st.flatten()

# plt.scatter(x1, y1, s=4)
# plt.scatter(x2, y2, s=4)
# plot_line(x1, y1)
# plot_line(x2, y2)
# plt.xlabel('pattern 1')
# plt.ylabel('consensus')
# plt.legend(['opinion', 'structure'])
# show_fig('corr_area')

# env index

g_index_mean_by_sim_active = np.mean(g_index_mean_active, axis=0, dtype=float)
g_index_op = g_index_mean_by_sim_active[..., 0]
g_index_st = g_index_mean_by_sim_active[..., 1]
g_index_diff = g_index_op - g_index_st

fig, (ax1, ax2, ax3) = plt_figure(n_col=3, hw_ratio=1, total_width=12)

cmap_arr5, cmap_setter5 = get_colormap([ax3], cmap='RdBu', fig=fig, vmin=-0.2, vmax=0.2, anchor='W')
cmap_arr4, cmap_setter4 = get_colormap([ax1, ax2], cmap='YlGnBu', vmin=0.4, vmax=0.8, seg=9, fig=fig, anchor='W')
  
ax1.imshow(g_index_st, **cmap_arr4)
ax2.imshow(g_index_op, **cmap_arr4)
ax3.imshow(g_index_diff, **cmap_arr5)

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

show_fig('env_index_heatmap')

# rest figures

pat_files_raw_op = pat_files_raw[::2]
pat_files_raw_st = pat_files_raw[1::2]

kde_gradation_cache = []

triads_cache = []

g_index_cache = []

for r in pat_files_raw_op, pat_files_raw_st:

  gradation, cluster, triads, in_degree, d_opinion, p_last, g_index_mean_by_rec_active = [
    np.array([x[k] for x in r]) \
      for k in ('pat_area_hp', 'cluster', 'triads', 'in_degree', 'opinion_diff', 'p_last', 'g_index_mean_active')
  ]
  
  in_degree_alpha, in_degree_p, in_degree_r = in_degree.T.copy()
  
  in_degree_bound = 10
  in_degree_alpha[in_degree_alpha > in_degree_bound] = in_degree_bound
  in_degree_alpha[in_degree_r <= 0] = in_degree_bound
  
  # is_consensus

  is_consensus = p_last < consensus_threshold
  is_not_consensus = np.logical_not(is_consensus)
  
  # is_near_diag
  
  rd_rate_vec_all = np.array([rd_rate_vec] * int(len(r) / rd_rate_vec.size)).flatten()
  is_near_diag = np.logical_and(rd_rate_vec_all > -1, rd_rate_vec_all < 1)
  is_not_near_diag = np.logical_not(is_near_diag)
  
  # gradation - consensus

  d = 0.005
  metrics = np.arange(0.3, 1 + d, d)
  kde_nc_raw = gaussian_kde(gradation[is_not_consensus])(metrics)
  kde_c_raw = gaussian_kde(gradation[is_consensus])(metrics)
  
  kde_all_raw = kde_nc_raw + kde_c_raw
  kde_all = kde_all_raw / (np.sum(kde_all_raw) * d)
  kde_nc = kde_nc_raw / (np.sum(kde_all_raw) * d)
  kde_c = kde_all - kde_nc
  kde_ratio_c = kde_c / kde_all
  
  kde_gradation_cache.append([metrics, kde_all, kde_nc, kde_c, kde_ratio_c])
  
  print(
    'gradation',
    np.mean(gradation[is_consensus]), 
    np.mean(gradation[is_not_consensus])
  )
  
  # env index
  # gradation - triads
  
  g_index_cache.append([
    gradation, 
    is_consensus, is_not_consensus,
    is_near_diag, is_not_near_diag,
    triads, g_index_mean_by_rec_active, 
  ])
  
  d_opinion[d_opinion < 0] = 0
  # plt.scatter(gradation, d_opinion, s=1)
  # print(plot_line(gradation[is_consensus], d_opinion[is_consensus]))
  # print(plot_line(gradation[is_not_consensus], d_opinion[is_not_consensus]))
  # show_fig()
  
  print(
    'd_opinion', 
    np.mean(d_opinion[is_consensus]), 
    np.mean(d_opinion[is_not_consensus])
  )

  # plt.hist(d_opinion[is_not_consensus])
  # plt.title('opinion peak distance')
  # show_fig()
  
  # plt.hist(in_degree_alpha)
  # plt.title('in-degree')
  # plt.xlabel(np.mean(in_degree_alpha))
  # show_fig()
  
  # plt.scatter(gradation[is_consensus], in_degree_alpha[is_consensus], s=1)
  # plt.scatter(gradation[is_not_consensus], in_degree_alpha[is_not_consensus], s=1)
  # show_fig()
  
  
  # plt.scatter(gradation[is_consensus], triads[is_consensus], s=.5, label='consensus')
  # plt.scatter(gradation[is_not_consensus], triads[is_not_consensus], s=.5, label='!consensus')
  # plt.title('#triads')
  # plt.legend()
  # show_fig()
  
# consensus

(grad_met_op, grad_all_op, grad_nc_op, grad_c_op, grad_ratio_c_op), \
(grad_met_st, grad_all_st, grad_nc_st, grad_c_st, grad_ratio_c_st) = kde_gradation_cache
fig, (axst, axop, axrt) = plt_figure(n_col = 3)

y_ticks = np.array([0, 1, 2, 4, 7, 12])

axop.plot(grad_met_op, grad_nc_op, label='polarized', color='tab:blue')
axop.plot(grad_met_op, grad_c_op, label='consented', color='tab:cyan')
axop.plot(grad_met_op, grad_all_op, label='all', color='tab:green')
axop.legend()

axst.plot(grad_met_st, grad_nc_st, label='polarized', color='tab:blue')
axst.plot(grad_met_st, grad_c_st, label='consented', color='tab:cyan')
axst.plot(grad_met_st, grad_all_st, label='all', color='tab:green')
axst.legend()

axrt.plot(grad_met_st, grad_ratio_c_st, label='structure')
axrt.plot(grad_met_op, grad_ratio_c_op, label='opinion')
axrt.legend()

axst.set_title('(a) structure', loc='left')
axop.set_title('(b) opinion', loc='left')
axrt.set_title('(c) %consented cases', loc='left')

axst.set_ylabel('probability')
for _ in (axst, axop, axrt):
  _.set_xlabel('gradation index')
axrt.set_ylabel('ratio')
  
for _ in (axst, axop):
  _.set_yticks(y_ticks)

show_fig('grad_consensus_rel')

# triads

def scatter_data(
  ax: Axes,
  x: NDArray, y: NDArray,
  c: NDArray, nc: NDArray,
  d: NDArray, nd: NDArray,
  s=4, lw=.8,
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
    
  return np.array([[np.mean(z[_i]) for _i in [_1, _2, _3, _4]] for z in [x, y]])

(g_op, c_op, nc_op, d_op, nd_op, tr_op, gi_op), \
  (g_st, c_st, nc_st, d_st, nd_st, tr_st, gi_st) = g_index_cache
fig, (axfreq, axst2, axop2) = plt_figure(n_col = 3, hw_ratio=4/5, total_width=18)

s = 3
lw = .5

kde_cl_op_ = gaussian_kde(tr_op)
kde_cl_st_ = gaussian_kde(tr_st)

metrics = np.arange(0, 40000, 100)
kde_cl_op = kde_cl_op_(metrics)
kde_cl_st = kde_cl_st_(metrics)

axfreq.plot(metrics, kde_cl_st, label='structure')
axfreq.plot(metrics, kde_cl_op, label='opinion')
axfreq.legend()

tr_st_stat = scatter_data(axst2, g_st, tr_st, c_st, nc_st, d_st, nd_st)
tr_op_stat = scatter_data(axop2, g_op, tr_op, c_op, nc_op, d_op, nd_op, legend=True)

tr_stat_diff = tr_op_stat - tr_st_stat

axfreq.set_title('(a) PDF of #C. T.', loc='left')
axst2.set_title('(b) structure', loc='left')
axop2.set_title('(c) opinion', loc='left')

axfreq.set_xlabel('#closed triads')
axfreq.set_ylabel('probability')

for _ in (axst2, axop2):
  _.set_ylabel('#closed triads')
  _.set_xlabel('gradation index')
  
plt.tight_layout()
show_fig('grad_triads_rel')


# env index

fig, (axfreq, axst2, axop2) = plt_figure(n_col = 3, hw_ratio=4/5, total_width=18)

kde_cl_op_ = gaussian_kde(gi_op)
kde_cl_st_ = gaussian_kde(gi_st)

metrics = np.arange(0.2, 1, 0.001)
kde_cl_op = kde_cl_op_(metrics)
kde_cl_st = kde_cl_st_(metrics)

axfreq.plot(metrics, kde_cl_st, label='structure')
axfreq.plot(metrics, kde_cl_op, label='opinion')
axfreq.legend()

gi_st_stat = scatter_data(axst2, g_st, gi_st, c_st, nc_st, d_st, nd_st)
gi_op_stat = scatter_data(axop2, g_op, gi_op, c_op, nc_op, d_op, nd_op, legend=True)

gi_stat_diff = gi_op_stat - gi_st_stat

axfreq.set_title('(a) PDF of environment index', loc='left')
axst2.set_title('(b) structure', loc='left')
axop2.set_title('(c) opinion', loc='left')

axfreq.set_xlabel('environment index')
axfreq.set_ylabel('probability')

for _ in (axst2, axop2):
  _.set_ylabel('environment index')
  _.set_xlabel('gradation index')
  
plt.tight_layout()
show_fig('grad_env_index_rel')

stat_diff_all = np.concatenate([tr_stat_diff, gi_stat_diff[1:]], axis=0)
numpy_to_latex_table(
  stat_diff_all, './fig_final/stat_diff_all.tex', 
  row_labels=['grad. index', '\#triads', 'env. index'], 
  col_labels=['ND;P', 'ND;C', 'D;P', 'D;C'])