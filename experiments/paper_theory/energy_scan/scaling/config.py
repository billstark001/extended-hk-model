"""Channel-scaling probe grids, numerical constants and input/output paths."""

from __future__ import annotations

import numpy as np

from experiments.paper_theory.paths import KINETIC as BINARY_PATH
from experiments.paper_theory.paths import MESO, output_root

HERE = output_root() / "energy_scan/scaling"
INPUT = HERE
GO_ROOT = MESO.resolve()
KINETIC = BINARY_PATH.resolve()
GRID_SIZE = 161
EPSILON = 0.45
MEAN_DEGREE = 15
RECOMMENDATION_COUNT = 10
ALPHA_PROBE = np.logspace(-3, 0, 13)
Q_PROBE = np.logspace(-3, 0, 10)
ALPHA_COLLAPSE = np.asarray((0.0025, 0.005, 0.01))
GAMMA_COLLAPSE = np.asarray((0.0, 2.0, 20.0))
TAU = np.linspace(0.0, 10.0, 51)
EPSILON_PROBE = np.linspace(0.05, 1.95, 77)
MASS_TOLERANCE = 1e-12
