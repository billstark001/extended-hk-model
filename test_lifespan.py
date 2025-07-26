

from result_interp.record import RawSimulationRecord
import works.config as cfg
from works.stat.context import c
import numpy as np

# key = 's_grad_sim1_rw7_dc0_rt3_op6'

key = 's_grad_sim9_rw7_dc0_rt3_op6'

# key = 's_grad_sim8_rw0_dc2_rt1_op6'
# key = 's_grad_sim8_rw0_dc0_rt1_op2'

rc1 = [x for x in cfg.all_scenarios_grad if x['UniqueName'] == key][0]

rec = RawSimulationRecord(
  cfg.get_workspace_dir(default='ssd_tmp'), rc1,
)


with rec:

  c.set_state(
      active_threshold=0.98,
      min_inactive_value=0.75,
      scenario_record=rec,
  )
  
  print(rec.max_step)
  
  c.debug = True
  
  retweeted_lifespans: np.ndarray = c.retweeted_lifespans
  retweeted_opinions: np.ndarray = c.retweeted_opinions

  print(retweeted_lifespans, retweeted_lifespans.size)
  
  
  c.gradation_index_hp