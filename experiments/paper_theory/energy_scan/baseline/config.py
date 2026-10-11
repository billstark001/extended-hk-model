"""Baseline physical/numerical grids and output paths; the CLI may override paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.paper_theory.paths import KINETIC as BINARY_PATH
from experiments.paper_theory.paths import MESO, SCAN_INPUT, output_root
from theory.runtime import kinetic
from theory.runtime.kinetic import KineticResolution, KineticStopping

HERE = output_root() / "energy_scan" / "baseline"
INPUT = HERE if "baseline" == "baseline" else SCAN_INPUT.resolve()
GO_ROOT = MESO.resolve()
KINETIC = BINARY_PATH.resolve()
ADAPTER = Path(kinetic.__file__)
ALPHA = 0.01
EPSILONS = np.asarray((0.2, 0.4, 0.8))
RATIOS = np.unique(
    np.concatenate(
        (np.expm1(np.linspace(0.0, np.log(101.0), 25)), np.asarray((1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)))
    )
)
RATIOS[np.isclose(RATIOS, 100.0, atol=1e-12, rtol=0.0)] = 100.0
RATIOS = np.unique(RATIOS)
PROGRESS_LEVELS = np.asarray((0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
ENERGY_NAMES = ("node_energy", "edge_energy")
GRID_SIZE = 161
MAX_STEPS = 10000
RECORD_EVERY = 1
WORKERS = 4
STOPPING = KineticStopping(
    mode="state_and_energy",
    minimum_steps=500,
    check_every=20,
    patience_steps=200,
    state_l1_tolerance=1e-07,
    energy_absolute_tolerance=1e-09,
    energy_relative_tolerance=1e-07,
)
RESOLUTION = KineticResolution()
