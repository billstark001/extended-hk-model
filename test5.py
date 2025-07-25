

from result_interp.record import RawSimulationRecord
import works.config as cfg
from works.stat.context import c
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from stats.distance_c import get_kde_pdf
from utils.plot import plot_network_snapshot

# key = 's_grad_sim1_rw7_dc0_rt3_op6'

key = 's_grad_sim9_rw7_dc0_rt1_op6'

# key = 's_grad_sim8_rw0_dc2_rt1_op6'
# key = 's_grad_sim8_rw0_dc0_rt1_op2'

rc1 = [x for x in cfg.all_scenarios_grad if x['UniqueName'] == key][0]

rec = RawSimulationRecord(
    cfg.get_workspace_dir(), rc1,
)


with rec:

  c.set_state(
      active_threshold=0.98,
      min_inactive_value=0.75,
      scenario_record=rec,
  )

  g = rec.get_graph(rec.max_step)
  opinion_last: np.ndarray = c.opinion_last

  plot_network_snapshot(None, opinion_last, g)
  plt.show()

  ls = np.linspace(0, 2, 1001)

  o_slice_mat = np.tile(opinion_last.reshape(
      (opinion_last.size, 1)), opinion_last.size)
  kde2 = get_kde_pdf(
      np.abs(o_slice_mat - o_slice_mat.T)
      [np.triu_indices(500, k=1)], 0.05, 0, 2
  )
  plt.plot(ls, kde2(ls))
