"""Parameter-clock quantile levels, plotting colors and input/output directories."""

import numpy as np

from experiments.paper_theory.release_clocks.artifact_paths import CLOCK_DATA
from experiments.paper_theory.release_clocks.artifact_paths import CLOCK_FIGURES as CLOCK_FIGURES
from experiments.paper_theory.release_clocks.artifact_paths import SOURCE as SOURCE

OUT = CLOCK_DATA
P = np.array([0.5, 0.9, 0.99])
COL = {"hk": "#2166ac", "deffuant": "#c42935"}
