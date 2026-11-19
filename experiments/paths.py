"""Path registrations for experiment inputs and outputs."""

from common.settings import register_path

SMP_BINARY = register_path("SMP_BINARY_PATH", "../social-media-models/smp")
SMP_BATCH_BINARY = register_path("SMP_BATCH_BINARY_PATH", "../social-media-models/smp-batch")
SMP_PROBE_BINARY = register_path("SMP_PROBE_BINARY", "../social-media-models/smp-probe")
WORKSPACE_MANIFEST = register_path("SIMULATION_WS_PATH", "sim_ws.json")
PAPER_OUTPUT = register_path("EHK_PAPER_OUTPUT_DIR", "outputs/experiments/paper")
PAPER_STAT_INPUT = register_path("SIMULATION_STAT_DIR", "outputs/experiments/paper")
