"""Expanded feasible-rate grids, scaled-time thresholds and input/output paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.paper_theory.paths import KINETIC as BINARY_PATH
from experiments.paper_theory.paths import MESO, SCAN_INPUT, output_root
from theory.runtime import kinetic
from theory.runtime.kinetic import KineticResolution

HERE = output_root() / "energy_scan" / "expanded"
INPUT = SCAN_INPUT.resolve()
GO_ROOT = MESO.resolve()
KINETIC = BINARY_PATH.resolve()
ADAPTER = Path(kinetic.__file__)
DYNAMICS = ("hk", "deffuant")
EPSILONS = np.asarray((0.2, 0.4, 0.8))
ALPHAS = 0.01 * np.power(2.0, np.arange(-4, 1, dtype=float) / 2.0)
POSITIVE_RATIOS = 100.0 * np.power(2.0, np.arange(-20, 5, dtype=float) / 2.0)
RATIOS = np.concatenate(([0.0], POSITIVE_RATIOS))
PROGRESS_LEVELS = np.asarray((0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
ENERGY_NAMES = ("node_energy", "edge_energy")
WELL_PROMINENCE_FRACTION = 0.02
GRID_SIZE = 161
MAX_TAU = 120.0
RECORD_TAU = 0.01
MINIMUM_TAU = 5.0
CHECK_TAU = 0.2
PATIENCE_TAU = 2.0
STATE_RATE_TOLERANCE_TAU = 1e-05
ENERGY_ABSOLUTE_TOLERANCE_TAU = 1e-07
ENERGY_RELATIVE_TOLERANCE_TAU = 1e-05
DEFAULT_WORKERS = 4
RESOLUTION = KineticResolution()
RAW = HERE / "raw_shards"
CHECKPOINTS = HERE / "case_checkpoints"
FIGURES = HERE / "figures"
EDGE_BETA_BOUNDS = (0.2, 5.0)
