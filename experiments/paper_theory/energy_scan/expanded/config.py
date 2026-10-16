"""Historical fit conventions; no scan grids, solver settings or output paths."""

import numpy as np

PROGRESS_LEVELS = np.asarray((0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
ENERGY_NAMES = ("node_energy", "edge_energy")
EDGE_BETA_BOUNDS = (0.2, 5.0)
