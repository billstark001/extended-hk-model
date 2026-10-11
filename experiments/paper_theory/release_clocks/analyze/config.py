"""Historical survival-fit quantile grids and generated table directories."""

import numpy as np

from ..artifact_paths import DATA as DATA
from ..artifact_paths import PILOT as PILOT
from ..artifact_paths import SOURCE as SOURCE

LEVELS = {
    "quantile3": np.array([0.1, 0.5, 0.9]),
    "quantile5": np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
    "quantile7": np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]),
    "quantile7_mid": np.array([0.05, 0.25, 0.5, 0.7, 0.85, 0.95, 0.99]),
    "quantile9_mid": np.array([0.05, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
    "quantile9": np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
}
