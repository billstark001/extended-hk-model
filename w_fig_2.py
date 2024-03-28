from typing import List, Dict, Union, Tuple
from numpy.typing import NDArray

import os
import importlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import seaborn as sns

from base.scenario import Scenario

from w_plot_utils import plt_figure
from w_scenarios import all_scenarios
import w_snapshots as ss

import w_proc_utils as p
import w_plot_utils as _p
importlib.reload(p)
importlib.reload(_p)

mpl.rcParams['font.size'] = 18
sns.set_theme(style='whitegrid')

BASE_PATH = './fig_final'

os.makedirs(BASE_PATH, exist_ok=True)


def plt_save_and_close(path: str):
  plt.savefig(path + '.eps', dpi=300, bbox_inches='tight')
  plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
  return plt.close()


def show_fig(name: str):
  plt_save_and_close(os.path.join(BASE_PATH, name))
  # plt.show()


def lim(
    start=0,
    end=1,
    pad=0.04
):
  duration = end - start
  left_pad = start - pad * duration
  right_pad = end + pad * duration
  return left_pad, right_pad


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

    sn, sr, an, ar, sum_n1, sum_r1, ratio_s, ratio_a = p.proc_opinion_diff(
        dn, dr, n_n, n_r,
        average
    )

    ratio_n, ratio_sum_log = p.proc_opinion_ratio(
        sum_n, sum_r, n_n, n_r
    )

    stats['o-step'] = steps
    stats['opinion'] = opinion
    stats['o-sn'] = sum_n1
    stats['o-sr'] = sum_r1
    stats['o-sn+sr'] = sum_n1 + sum_r1
    stats['o-an'] = an
    stats['o-ar'] = ar
    stats['o-an+ar'] = an + ar
    stats['ratio-s'] = ratio_s
    stats['ratio-a'] = ratio_a
    stats['ratio-n'] = ratio_n
    stats['ratio-sum-log'] = ratio_sum_log

    all_data.append(stats)

    print(scenario_name)

S_rr, S_ro, S_rs, S_rm3, S_rm7, S_sfr, S_sfo, S_sfs, S_sfm3, S_sfm7 = all_data
print('Scenarios Loaded.')


# data

for S in all_data:
  S['o-cluster-reached'] = S['step'][p.first_index_above_min(
    np.array(S['distance-worst-o']),
    1e-3
  )]
  

# plots


# triads & c-coeff.

fig, axes = plt_figure(n_col=4)
axtr, axts, axcr, axcs = axes

for ax in [axtr, axts]:
  ax.set_yscale('log')
  ax.grid(True, linestyle='dashed', which='both')

axtr.plot(S_rr['step'], S_rr['triads'])
axtr.plot(S_rs['step'], S_rs['triads'])
axtr.plot(S_ro['step'], S_ro['triads'])
axtr.plot(S_rm3['step'], S_rm3['triads'])
axtr.plot(S_rm7['step'], S_rm7['triads'])

axtr.set_title('(a) random', loc='left')
axtr.set_ylabel('#closed triads')


axts.plot(S_sfr['step'], S_sfr['triads'])
axts.plot(S_sfs['step'], S_sfs['triads'])
axts.plot(S_sfo['step'], S_sfo['triads'])
axts.plot(S_sfm3['step'], S_sfm3['triads'])
axts.plot(S_sfm7['step'], S_sfm7['triads'])

axts.set_title('(b) scale-free', loc='left')


axcr.plot(S_rr['step'], S_rr['cluster'])
axcr.plot(S_rs['step'], S_rs['cluster'])
axcr.plot(S_ro['step'], S_ro['cluster'])
axcr.plot(S_rm3['step'], S_rm3['cluster'])
axcr.plot(S_rm7['step'], S_rm7['cluster'])

axcr.set_title('(c) random', loc='left')
axcr.set_ylabel('clustering coefficient')


axcs.plot(S_sfr['step'], S_sfr['cluster'])
axcs.plot(S_sfs['step'], S_sfs['cluster'])
axcs.plot(S_sfo['step'], S_sfo['cluster'])
axcs.plot(S_sfm3['step'], S_sfm3['cluster'])
axcs.plot(S_sfm7['step'], S_sfm7['cluster'])

axcs.legend(['random', 'structure', 'opinion', 'mixed-3', 'mixed-7'])
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
            * 2, lw=1, linestyle='dashed', color='black')
  ax_a.plot(stats_index, np.ones(len(stats_index))
            * 3, lw=1, linestyle='dashed', color='gray')

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
            * 0.05, lw=1, linestyle='dashed', color='gray')
  ax_p.plot(stats_index, np.ones(len(stats_index))
            * 0.01, lw=1, linestyle='dashed', color='black')


print_power_law_data(axrra, axrrp, dict(_=S_rr))
print_power_law_data(axroa, axrop, dict(_=S_ro))
print_power_law_data(axrsa, axrsp, {
    'structure': S_rs,
    'mixed-3': S_rm3,
    'mixed-7': S_rm7
})
print_power_law_data(axsfa, axsfp, dict(
    random=S_sfr, structure=S_sfs, opinion=S_sfo, **({
        'mixed-3': S_sfm3,
        'mixed-7': S_sfm7,
    })))

axrra.set_ylabel('\\alpha')
axrrp.set_ylabel('p-value')

axrra.set_title('(a) random, random', loc='left')
axroa.set_title('(b) random, opinion', loc='left')
axrsa.set_title('(c) random, structure', loc='left')
axsfa.set_title('(d) scale-free networks', loc='left')

show_fig('power-law')


# opinion contributions

fig, _axes = plt_figure(n_row=3, n_col=3, sharey='row',
                        sharex='col', total_width=10)
axes: List[List[Axes]] = _axes
(axrro, axrso, axroo), (axrrc, axrsc, axroc), (axrrs, axrss, axros) = axes


def print_contrib_data(ax_o: Axes, ax_c: Axes, ax_s: Axes, stats: StatsType, legend=False):

  ax_o.plot(stats['o-step'], stats['opinion'], lw=0.2)
  ax_c.plot(stats['o-step'], stats['o-sn'])
  ax_c.plot(stats['o-step'], stats['o-sr'])
  ax_c.plot(stats['o-step'], stats['o-sn+sr'])
  ax_s.plot(stats['o-step'], stats['o-an'])
  ax_s.plot(stats['o-step'], stats['o-ar'])
  ax_s.plot(stats['o-step'], stats['o-an+ar'])
  if legend:
    ax_s.legend(['followed', 'recommended', 'total'])


print_contrib_data(axrro, axrrc, axrrs, S_rr)
print_contrib_data(axrso, axrsc, axrss, S_rs)
print_contrib_data(axroo, axroc, axros, S_ro, legend=True)

axrro.set_title('(a) random, random', loc='left')
axrso.set_title('(b) random, structure', loc='left')
axroo.set_title('(c) random, opinion', loc='left')

axrro.set_ylabel('opinion')
axrrc.set_ylabel('#concordant users')
axrrs.set_ylabel('absolute contributions')

scales = [1, 1, 4]
for _ in axes:
  for i, __ in enumerate(_):
    scale = scales[i]
    __.set_xlim(left=-10 * scale, right=(10 + 250) * scale)

plt.subplots_adjust(wspace=0.06, hspace=0.1)
show_fig('opinion-contrib')


# distance distribution


axes: List[Axes]
fig, axes = plt_figure(n_col=4, sharey='row')
axro, axsfo, axrs, axsfs = axes

axro.plot(S_rr['step'], S_rr['distance-worst-o'])
axro.plot(S_rs['step'], S_rs['distance-worst-o'])
axro.plot(S_ro['step'], S_ro['distance-worst-o'])
axro.plot(S_rm3['step'], S_rm3['distance-worst-o'])
axro.plot(S_rm7['step'], S_rm7['distance-worst-o'])

axsfo.plot(S_sfr['step'], S_sfr['distance-worst-o'])
axsfo.plot(S_sfs['step'], S_sfs['distance-worst-o'])
axsfo.plot(S_sfo['step'], S_sfo['distance-worst-o'])
axsfo.plot(S_sfm3['step'], S_sfm3['distance-worst-o'])
axsfo.plot(S_sfm7['step'], S_sfm7['distance-worst-o'])

axrs.plot(S_rr['step'], S_rr['distance-worst-s'])
axrs.plot(S_rs['step'], S_rs['distance-worst-s'])
axrs.plot(S_ro['step'], S_ro['distance-worst-s'])
axrs.plot(S_rm3['step'], S_rm3['distance-worst-s'])
axrs.plot(S_rm7['step'], S_rm7['distance-worst-s'])

axsfs.plot(S_sfr['step'], S_sfr['distance-worst-s'])
axsfs.plot(S_sfs['step'], S_sfs['distance-worst-s'])
axsfs.plot(S_sfo['step'], S_sfo['distance-worst-s'])
axsfs.plot(S_sfm3['step'], S_sfm3['distance-worst-s'])
axsfs.plot(S_sfm7['step'], S_sfm7['distance-worst-s'])

axsfs.legend(['random', 'structure', 'opinion', 'mixed-3', 'mixed-7'])

axro.set_title('(a) random, objective', loc='left')
axsfo.set_title('(b) scale-free, objective', loc='left')
axrs.set_title('(c) random, subjective', loc='left')
axsfs.set_title('(d) scale-free, subjective', loc='left')

axro.set_ylabel('rel. distance to worst distribution')

plt.subplots_adjust(wspace=0.06, hspace=0.15)
show_fig('dist-of-dis')


# violin
violin_x_1 = [
    ('random', S_rr),
    ('structure', S_rs),
    ('mixed-3', S_rm3),
    ('mixed-7', S_rm7),
    ('opinion', S_ro),
]
violin_x_2 = [
    ('random', S_sfr),
    ('structure', S_sfs),
    ('mixed-3', S_sfm3),
    ('mixed-7', S_sfm7),
    ('opinion', S_sfo),
]

violin_x_axis = np.arange(1, len(violin_x_1) + 1)
violin_x_name = [x[0] for x in violin_x_1]

axes: List[Axes]
fig, axes = plt_figure(
    n_col=2, n_row=2, sharex='col',
    sharey='row', total_width=7, hw_ratio=1)
(axup_r, axup_sf), (axdn_r, axdn_sf) = axes

axup_r.violinplot([x[1]['ratio-n'] for x in violin_x_1], showmeans=True)
axdn_r.violinplot([x[1]['ratio-sum-log'] for x in violin_x_1], showmeans=True)

axup_sf.violinplot([x[1]['ratio-n'] for x in violin_x_2], showmeans=True)
axdn_sf.violinplot([x[1]['ratio-sum-log'] for x in violin_x_2], showmeans=True)


axup_r.set_title('(d) random', loc='left')
axup_sf.set_title('(e) scale-free', loc='left')
axup_r.set_ylabel('ratio of concordant recommendation')
axdn_r.set_ylabel('contribution of recommendation')

axdn_r.set_xticks(violin_x_axis, violin_x_name, rotation=90)
axdn_sf.set_xticks(violin_x_axis, violin_x_name, rotation=90)

ylim_s, ylim_e = lim(0.1, 0.6)
axup_r.set_ylim(bottom=ylim_s, top=ylim_e)
ylim_s, ylim_e = lim(-0.6, 0.4)
axdn_r.set_ylim(bottom=ylim_s, top=ylim_e)

plt.subplots_adjust(wspace=0.06, hspace=0.05)
show_fig('violin')
