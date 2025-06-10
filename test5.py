

from result_interp.record import RawSimulationRecord
import works.config as cfg
from works.stat.context import c

idx =  28 * 16 + 1 # rt0, op0

rc1 = cfg.all_scenarios_grad[idx]

rec = RawSimulationRecord(
  cfg.SIMULATION_RESULT_DIR, rc1,
)

rec.load()

c.set_state(
    active_threshold=0.98,
    min_inactive_value=0.75,
    scenario_record=rec,
)

_ = c.gradation_index_hp
_ = c.x_mean_vars

