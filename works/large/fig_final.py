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

import utils.stat as p
import utils.plot as _p
importlib.reload(p)
importlib.reload(_p)

from utils.plot import plt_figure
from works.large.scenarios import all_scenarios
import works.large.snapshots as ss

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

S_sr1, S_or1, S_sr2, S_or2, S_sr3, S_or3, S_sr4, S_or4, S_sr5, S_or5,\
  S_ss1, S_os1, S_ss2, S_os2, S_ss3, S_os3, S_ss4, S_os4, S_ss5, S_os5, = all_data
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

axtr.plot(S_sr1['step'], S_sr1['triads'])
axtr.plot(S_sr2['step'], S_sr2['triads'])
axtr.plot(S_or1['step'], S_or1['triads'])
axtr.plot(S_or2['step'], S_or2['triads'])
axtr.plot(S_sr3['step'], S_sr3['triads'])

axtr.set_title('(a) random', loc='left')
axtr.set_ylabel('#closed triads')


axts.plot(S_or3['step'], S_or3['triads'])
axts.plot(S_or4['step'], S_or4['triads'])
axts.plot(S_sr4['step'], S_sr4['triads'])
axts.plot(S_sr5['step'], S_sr5['triads'])
axts.plot(S_or5['step'], S_or5['triads'])

axts.set_title('(b) scale-free', loc='left')


axcr.plot(S_sr1['step'], S_sr1['cluster'])
axcr.plot(S_sr2['step'], S_sr2['cluster'])
axcr.plot(S_or1['step'], S_or1['cluster'])
axcr.plot(S_or2['step'], S_or2['cluster'])
axcr.plot(S_sr3['step'], S_sr3['cluster'])

axcr.set_title('(c) random', loc='left')
axcr.set_ylabel('clustering coefficient')


axcs.plot(S_or3['step'], S_or3['cluster'])
axcs.plot(S_or4['step'], S_or4['cluster'])
axcs.plot(S_sr4['step'], S_sr4['cluster'])
axcs.plot(S_sr5['step'], S_sr5['cluster'])
axcs.plot(S_or5['step'], S_or5['cluster'])

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


print_power_law_data(axrra, axrrp, dict(_=S_sr1))
print_power_law_data(axroa, axrop, dict(_=S_or1))
print_power_law_data(axrsa, axrsp, {
    'structure': S_sr2,
    'mixed-3': S_or2,
    'mixed-7': S_sr3
})
print_power_law_data(axsfa, axsfp, dict(
    random=S_or3, structure=S_or4, opinion=S_sr4, **({
        'mixed-3': S_sr5,
        'mixed-7': S_or5,
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


print_contrib_data(axrro, axrrc, axrrs, S_sr1)
print_contrib_data(axrso, axrsc, axrss, S_sr2)
print_contrib_data(axroo, axroc, axros, S_or1, legend=True)

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

axro.plot(S_sr1['step'], S_sr1['distance-worst-o'])
axro.plot(S_sr2['step'], S_sr2['distance-worst-o'])
axro.plot(S_or1['step'], S_or1['distance-worst-o'])
axro.plot(S_or2['step'], S_or2['distance-worst-o'])
axro.plot(S_sr3['step'], S_sr3['distance-worst-o'])

axsfo.plot(S_or3['step'], S_or3['distance-worst-o'])
axsfo.plot(S_or4['step'], S_or4['distance-worst-o'])
axsfo.plot(S_sr4['step'], S_sr4['distance-worst-o'])
axsfo.plot(S_sr5['step'], S_sr5['distance-worst-o'])
axsfo.plot(S_or5['step'], S_or5['distance-worst-o'])

axrs.plot(S_sr1['step'], S_sr1['distance-worst-s'])
axrs.plot(S_sr2['step'], S_sr2['distance-worst-s'])
axrs.plot(S_or1['step'], S_or1['distance-worst-s'])
axrs.plot(S_or2['step'], S_or2['distance-worst-s'])
axrs.plot(S_sr3['step'], S_sr3['distance-worst-s'])

axsfs.plot(S_or3['step'], S_or3['distance-worst-s'])
axsfs.plot(S_or4['step'], S_or4['distance-worst-s'])
axsfs.plot(S_sr4['step'], S_sr4['distance-worst-s'])
axsfs.plot(S_sr5['step'], S_sr5['distance-worst-s'])
axsfs.plot(S_or5['step'], S_or5['distance-worst-s'])

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
    ('random', S_sr1),
    ('structure', S_sr2),
    ('mixed-3', S_or2),
    ('mixed-7', S_sr3),
    ('opinion', S_or1),
]
violin_x_2 = [
    ('random', S_or3),
    ('structure', S_or4),
    ('mixed-3', S_sr5),
    ('mixed-7', S_or5),
    ('opinion', S_sr4),
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
