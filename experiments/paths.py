"""Path registrations for experiment inputs and outputs."""

from common.settings import register_path

SMP_BINARY = register_path("SMP_BINARY_PATH", "../social-media-models/smp")
WORKSPACE_MANIFEST = register_path("SIMULATION_WS_PATH", "sim_ws.json")
PAPER_OUTPUT = register_path("EHK_PAPER_OUTPUT_DIR", "outputs/experiments/paper")
PAPER_STAT_INPUT = register_path("SIMULATION_STAT_DIR", "outputs/experiments/paper")
