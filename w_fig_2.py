from typing import List, Dict, Union, Tuple
from numpy.typing import NDArray

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import seaborn as sns

from base.scenario import Scenario

from w_scenarios import all_scenarios
import w_snapshots as ss
import w_proc_utils as p

mpl.rcParams['font.size'] = 18
sns.set_theme(style='whitegrid')

BASE_PATH = './fig_final'

os.makedirs(BASE_PATH, exist_ok=True)


def plt_figure(n_row=1, n_col=1, hw_ratio=3/4, total_width=16, **kwargs) -> Tuple[Figure, List[Axes]]:
  width = total_width / n_col
  height = width * hw_ratio
  total_height = height * n_row
  return plt.subplots(n_row, n_col, figsize=(total_width, total_height), **kwargs)


def plt_save_and_close(path: str):
  plt.savefig(path + '.eps', dpi=300, bbox_inches='tight')
  plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
  return plt.close()


def show_fig(name: str):
  plt_save_and_close(os.path.join(BASE_PATH, name))
  # plt.show()

def moving_average(data: NDArray, window_size: int):
  if window_size < 2:
    return data
  pad_width = window_size // 2
  pad_data = np.pad(data, pad_width, mode='edge')
  window = np.ones(window_size) / window_size
  moving_avg = np.convolve(pad_data, window, 'valid')
  return moving_avg

# load data


StatsType = Dict[str, Union[List[float], NDArray]]
all_data: List[StatsType]
try:
  assert len(all_data) > 0
except:
  all_data = []


average = 3
if len(all_data) == 0:
  for scenario_name, scenario_params in all_scenarios.items():

    # load model
    model = Scenario(*scenario_params)
    snapshot, snapshot_name = ss.load_latest(scenario_name)
    if not snapshot:
      continue
    model.load(*snapshot)

    stats = model.generate_stats()
    steps, opinion, dn, dr, sum_n, sum_r, n_n, n_r = model.get_opinion_data()
    dn[0] = dn[1]
    dr[0] = dr[1]
    
    sn, sr, an, ar, ratio_s, ratio_a = p.proc_opinion_diff(dn, dr, average)

    stats['o-step'] = steps
    stats['opinion'] = opinion
    stats['o-sn'] = sn
    stats['o-sr'] = sr
    stats['o-sn+sr'] = sn + sr
    stats['o-an'] = an
    stats['o-ar'] = ar
    stats['o-an+ar'] = an + ar
    stats['ratio-s'] = ratio_s
    stats['ratio-a'] = ratio_a

    all_data.append(stats)

    print(scenario_name)

S_rr, S_ro, S_rs, S_sfr, S_sfo, S_sfs = all_data
print('Scenarios Loaded.')

# plots


# triads & c-coeff.

fig, axes = plt_figure(n_col=4)
axtr, axts, axcr, axcs = axes

for ax in axes:
  ax.set_yscale('log')
  ax.grid(True, linestyle='dashed', which='both')

axtr.plot(S_rr['step'], S_rr['triads'])
axtr.plot(S_ro['step'], S_ro['triads'])
axtr.plot(S_rs['step'], S_rs['triads'])

axtr.set_title('(a) random', loc='left')
axtr.set_ylabel('#closed triads')


axts.plot(S_sfr['step'], S_sfr['triads'])
axts.plot(S_sfo['step'], S_sfo['triads'])
axts.plot(S_sfs['step'], S_sfs['triads'])

axts.set_title('(b) scale-free', loc='left')


axcr.plot(S_rr['step'], S_rr['cluster'])
axcr.plot(S_ro['step'], S_ro['cluster'])
axcr.plot(S_rs['step'], S_rs['cluster'])

axcr.set_title('(c) random', loc='left')
axcr.set_ylabel('clustering coefficient')


axcs.plot(S_sfr['step'], S_sfr['cluster'])
axcs.plot(S_sfo['step'], S_sfo['cluster'])
axcs.plot(S_sfs['step'], S_sfs['cluster'])

axcs.legend(['random', 'opinion', 'structure'])
axcs.set_title('(d) scale-free', loc='left')

plt.subplots_adjust(wspace=0.3, hspace=0.2)
show_fig('triads-c-coeff')


# power-law related

fig, _axes = plt_figure(n_row=2, n_col=4)
axes: List[List[Axes]] = _axes
(axrra, axroa, axrsa, axsfa), (axrrp, axrop, axrsp, axsfp) = axes


def print_power_law_data(ax_a: Axes, ax_p: Axes, all_stats: Dict[str, StatsType]):

  max_x = 0
  legend = []
  for name, stats in all_stats.items():
    max_x = max(max_x, np.max(stats['step']))
    legend.append(name)

  stats_index = np.array([0, max_x])

  for stats in all_stats.values():
    ax_a.plot(stats['step'], stats['in-degree-alpha'], lw=1.5)
  if len(legend) > 1:
    ax_a.legend(legend)
  ax_a.plot(stats_index, np.ones(len(stats_index))
            * 2, lw=1, linestyle='dashed')
  ax_a.plot(stats_index, np.ones(len(stats_index))
            * 3, lw=1, linestyle='dashed')

  for stats in all_stats.values():
    ax_p.plot(stats['step'], stats['in-degree-p-value'])
  if len(legend) > 1:
    ax_p.legend(legend)
  
  for stats in all_stats.values():
    stats_R = np.array(stats['in-degree-R'])
    stats_p = np.array(stats['in-degree-p-value'])
    stats_step = np.array(stats['step'])
    mask = stats_R <= 0
    ax_p.scatter(stats_step[mask], stats_p[mask], marker='x', color='red')
    
  ax_p.plot(stats_index, np.ones(len(stats_index))
            * 0.05, lw=1, linestyle='dashed')
  ax_p.plot(stats_index, np.ones(len(stats_index))
            * 0.01, lw=1, linestyle='dashed')


print_power_law_data(axrra, axrrp, dict(_=S_rr))
print_power_law_data(axroa, axrop, dict(_=S_ro))
print_power_law_data(axrsa, axrsp, dict(_=S_rs))
print_power_law_data(axsfa, axsfp, dict(
    random=S_sfr, opinion=S_sfo, structure=S_sfs))

axrra.set_ylabel('\\alpha')
axrrp.set_ylabel('p-value')

axrra.set_title('(a) random, random', loc='left')
axroa.set_title('(b) random, opinion', loc='left')
axrsa.set_title('(c) random, structure', loc='left')
axsfa.set_title('(d) scale-free networks', loc='left')

show_fig('power-law')


# opinion contributions

fig, _axes = plt_figure(n_row=2, n_col=3, sharey='row')
axes: List[List[Axes]] = _axes
(axrro, axroo, axrso), (axrrc, axroc, axrsc) = axes


def print_contrib_data(ax_o: Axes, ax_c: Axes, stats: StatsType):

  ax_o.plot(stats['o-step'], stats['opinion'], lw=0.2)
  ax_c.plot(stats['o-step'], stats['o-sn'])
  ax_c.plot(stats['o-step'], stats['o-sr'])
  ax_c.plot(stats['o-step'], stats['o-sn+sr'])
  ax_c.legend(['followed', 'recommended', 'total'])


print_contrib_data(axrro, axrrc, S_rr)
print_contrib_data(axroo, axroc, S_ro)
print_contrib_data(axrso, axrsc, S_rs)

axrro.set_title('(a) random, random', loc='left')
axroo.set_title('(b) random, opinion', loc='left')
axrso.set_title('(c) random, structure', loc='left')

axrro.set_ylabel('opinion')
axrrc.set_ylabel('contributions')

scales = [1, 4, 1]
for _ in axes:
  for i, __ in enumerate(_):
    scale = scales[i]
    __.set_xlim(left=-10 * scale , right=(10 + 250) * scale)

plt.subplots_adjust(wspace=0.06, hspace=0.15)
show_fig('opinion-contrib')


# distance distribution


axes: List[Axes]
fig, axes = plt_figure(n_col=4, sharey='row')
axro, axsfo, axrs, axsfs = axes

axro.plot(S_rr['step'], S_rr['distance-worst-o'])
axro.plot(S_ro['step'], S_ro['distance-worst-o'])
axro.plot(S_rs['step'], S_rs['distance-worst-o'])

axsfo.plot(S_sfr['step'], S_sfr['distance-worst-o'])
axsfo.plot(S_sfo['step'], S_sfo['distance-worst-o'])
axsfo.plot(S_sfs['step'], S_sfs['distance-worst-o'])

axrs.plot(S_rr['step'], S_rr['distance-worst-s'])
axrs.plot(S_ro['step'], S_ro['distance-worst-s'])
axrs.plot(S_rs['step'], S_rs['distance-worst-s'])

axsfs.plot(S_sfr['step'], S_sfr['distance-worst-s'])
axsfs.plot(S_sfo['step'], S_sfo['distance-worst-s'])
axsfs.plot(S_sfs['step'], S_sfs['distance-worst-s'])
axsfs.legend(['random', 'opinion', 'structure'])

axro.set_title('(a) random, objective', loc='left')
axsfo.set_title('(b) scale-free, objective', loc='left')
axrs.set_title('(c) random, subjective', loc='left')
axsfs.set_title('(d) scale-free, subjective', loc='left')

axro.set_ylabel('rel. distance to worst distribution')

plt.subplots_adjust(wspace=0.06, hspace=0.15)
show_fig('dist-of-dis')