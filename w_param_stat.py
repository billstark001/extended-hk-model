from typing import cast, List, Iterable, Union

import os
import pickle
import json
import importlib

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from base import Scenario

import w_param_search as p
import w_param_plot as pp

import w_plot_utils as _p
importlib.reload(_p)
importlib.reload(p)
importlib.reload(pp)

from w_plot_utils import plot_network_snapshot, plt_figure, get_colormap

# parameters

plot_path = './fig2'
pat_file_paths = [
  './fig2/pattern_stats.json',
  './fig2/pattern_stats_2.json',
]

mpl.rcParams['font.size'] = 18


BASE_PATH = './fig_final'

os.makedirs(BASE_PATH, exist_ok=True)


def plt_save_and_close(path: str):
  plt.savefig(path + '.eps', dpi=300, bbox_inches='tight')
  plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
  return plt.close()


def show_fig(name: str = 'test'):
  # plt_save_and_close(os.path.join(BASE_PATH, name))
  plt.show()
  
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
  

keys = ['name', 'active_step', 'pat_abs_mean', 'pat_area_hp', 'p_last']
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
names, active_steps, abs_hp, areas_hp, hs_last = pat_csv_values

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



# figure 1

fig, ((ax1, ax2), (ax3, ax4)) = plt_figure(n_col=2, n_row=2, hw_ratio=4/5)

cmap='YlGnBu'
cmap_arr5, cmap_setter5 = get_colormap([ax3, ax4], cmap=cmap, fig=fig, vmin=-0.3, vmax=0.3)
cmap_arr4, cmap_setter4 = get_colormap([ax1, ax2], cmap=cmap, vmin=0.3, seg=8, fig=fig)
  
ax1.imshow(m_pattern_1_a_st, **cmap_arr4)
ax2.imshow(m_pattern_1_a_op, **cmap_arr4)
ax3.imshow(m_pattern_1_a_op - m_pattern_1_a_st, **cmap_arr5)
ax4.imshow(m_pattern_1_a_op - m_pattern_1_a_st > 0)

for _ in (ax1, ax2, ax3, ax4):
  _.invert_yaxis()
  _.set_xticks(np.arange(p.decay_rate_array.size))
  _.set_yticks(np.arange(p.rewiring_rate_array.size))
  _.set_xticklabels(p.decay_rate_array)
  _.set_yticklabels(p.rewiring_rate_array)

ax1.set_title('structure')
ax2.set_title('opinion')
ax3.set_title('o - s')
ax4.set_title('o - s > 0')

ax3.set_xlabel('decay')
ax3.set_ylabel('rewiring')

cmap_setter4()
cmap_setter5()
show_fig('pat1_area')

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

x1, y1 = m_pattern_1_a_op.flatten(), m_is_consensus_op.flatten()
x2, y2 = m_pattern_1_a_st.flatten(), m_is_consensus_st.flatten()

plt.scatter(x1, y1, s=4)
plt.scatter(x2, y2, s=4)
plot_line(x1, y1)
plot_line(x2, y2)
plt.xlabel('pattern 1')
plt.ylabel('consensus')
plt.legend(['opinion', 'structure'])
show_fig('corr_area')


# figure 3

pat_files_raw_op = pat_files_raw[::2]
pat_files_raw_st = pat_files_raw[1::2]

for r in pat_files_raw_op, pat_files_raw_st:

  gradation, cluster, triads, in_degree, d_opinion, p_last = [
    np.array([x[k] for x in r]) \
      for k in ('pat_area_hp', 'cluster', 'triads', 'in_degree', 'opinion_diff', 'p_last')
  ]
  
  in_degree_alpha, in_degree_p, in_degree_r = in_degree.T.copy()
  
  in_degree_bound = 10
  in_degree_alpha[in_degree_alpha > in_degree_bound] = in_degree_bound
  in_degree_alpha[in_degree_r <= 0] = in_degree_bound

  is_consensus = p_last < consensus_threshold
  is_not_consensus = np.logical_not(is_consensus)

  plt.hist(gradation[np.logical_not(is_consensus)])
  plt.hist(gradation[is_consensus])
  show_fig('test1')
  
  print(
    'gradation',
    np.mean(gradation[is_consensus]), 
    np.mean(gradation[is_not_consensus])
  )
  
  
  d_opinion[d_opinion < 0] = 0
  plt.scatter(gradation, d_opinion, s=1)
  print(plot_line(gradation[is_consensus], d_opinion[is_consensus]))
  print(plot_line(gradation[is_not_consensus], d_opinion[is_not_consensus]))
  show_fig()
  
  print(
    'd_opinion', 
    np.mean(d_opinion[is_consensus]), 
    np.mean(d_opinion[is_not_consensus])
  )

  plt.hist(d_opinion[is_not_consensus])
  plt.title('opinion peak distance')
  show_fig()
  
  # plt.hist(in_degree_alpha)
  # plt.title('in-degree')
  # plt.xlabel(np.mean(in_degree_alpha))
  # show_fig()
  
  # plt.scatter(gradation[is_consensus], in_degree_alpha[is_consensus], s=1)
  # plt.scatter(gradation[is_not_consensus], in_degree_alpha[is_not_consensus], s=1)
  # show_fig()
  
  plt.scatter(gradation[is_consensus], cluster[is_consensus], s=.5, label='consensus')
  plt.scatter(gradation[is_not_consensus], cluster[is_not_consensus], s=.5, label='!consensus')
  plt.title('clustering coeff.')
  plt.legend()
  show_fig()
  
  
  # plt.scatter(gradation[is_consensus], triads[is_consensus], s=.5, label='consensus')
  # plt.scatter(gradation[is_not_consensus], triads[is_not_consensus], s=.5, label='!consensus')
  # plt.title('#triads')
  # plt.legend()
  # show_fig()
  
  