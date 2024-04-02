from typing import cast, List

import os
import pickle
import importlib

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from base import Scenario

import w_param_search as p
import w_param_plot as pp
from w_plot_utils import plot_network_snapshot, plt_figure

import w_plot_utils as _p
importlib.reload(_p)
importlib.reload(p)
importlib.reload(pp)

# parameters

plot_path = './fig2'
pat_csv_path = './fig2/pattern_stats.csv'

mpl.rcParams['font.size'] = 18

# prepare data

pat_csv = pd.read_csv(pat_csv_path, encoding='utf-8-sig')
pat_csv_values_raw = pat_csv.values

total_len, n_stat_data = pat_csv_values_raw.shape
full_sim_len = p.rewiring_rate_array.shape[0] * p.decay_rate_array.shape[0] * len(p.n_gens)

used_len = total_len - total_len % full_sim_len

pat_csv_values = pat_csv_values_raw[:used_len].T.reshape((
  n_stat_data,
  p.n_sims,
  p.rewiring_rate_array.shape[0],
  p.decay_rate_array.shape[0],
  len(p.n_gens),
))
# axes: (#sim, rewiring, decay, recsys)
names, steps, active_steps, means, stds, hs_last = pat_csv_values

# in the following operations, average data is calculated in all simulations

m_active_step = np.mean(active_steps, axis=0).astype(float)
m_pattern_1 = np.mean(means, axis=0).astype(float)
m_pattern_1_std = np.mean(stds, axis=0).astype(float)

consensus_threshold = 0.6
m_hs_last = np.mean(hs_last, axis=0).astype(float)
m_is_consensus = np.mean(hs_last < consensus_threshold, axis=0).astype(float)

# m_pattern_1_op = means[..., 0, :].astype(float)
# m_pattern_1_st = means[..., 1, :].astype(float)
# m_hs_last_op = hs_last[..., 0, :].astype(float)
# m_hs_last_st = hs_last[..., 1, :].astype(float)


m_pattern_1_op = m_pattern_1[..., 0]
m_pattern_1_st = m_pattern_1[..., 1]
m_hs_last_op = m_hs_last[..., 0]
m_hs_last_st = m_hs_last[..., 1]

# figure 1

fig, ((ax1, ax2), (ax3, ax4)) = plt_figure(n_col=2, n_row=2, hw_ratio=4/5)


cmap='YlGnBu'
cmap_arr = dict(cmap=cmap, vmin=-1, vmax=1)
  
ax1.imshow(m_pattern_1_st, **cmap_arr)
ax2.imshow(m_pattern_1_op, **cmap_arr)
ax3.imshow(m_pattern_1_op - m_pattern_1_st, **cmap_arr)
ax4.imshow(m_pattern_1_op - m_pattern_1_st > 0, **cmap_arr)

for _ in (ax1, ax2, ax3, ax4):
  _.set_xticks(np.arange(6))
  _.set_yticks(np.arange(6))
  _.set_xticklabels(p.decay_rate_array)
  _.set_yticklabels(p.rewiring_rate_array)

ax1.set_title('structure')
ax2.set_title('opinion')
ax3.set_title('o - s')
ax4.set_title('o - s > 0')

ax3.set_xlabel('decay')
ax3.set_ylabel('rewiring')


norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ticks=np.linspace(-1, 1, 5), ax=[ax2, ax4])

plt.show()

# figure 2

x1, y1 = m_pattern_1_op.flatten(), 1 - m_hs_last_op.flatten()
x2, y2 = m_pattern_1_st.flatten(), 1 - m_hs_last_st.flatten()

def linear_func(x, a, b):
  return a * x + b

(a1, b1), _ = curve_fit(linear_func, x1, y1)
(a2, b2), _ = curve_fit(linear_func, x2, y2)

plt.scatter(x1, y1, s=4)
plt.scatter(x2, y2, s=4)
plt.plot(x1, a1 * x1 + b1)
plt.plot(x2, a2 * x2 + b2)
plt.xlabel('pattern 1')
plt.ylabel('consensus')
plt.legend(['opinion', 'structure'])
plt.show()