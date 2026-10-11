"""Read-only inputs and separate generated outputs; no import-time writes."""

import os
from pathlib import Path

from ..paths import CLOCK_INPUT, SCAN_INPUT, output_root


def configured(name, default):
    """Resolve an environment path or fallback without creating it."""
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


ROOT = configured("EHK_CLOCK_OUTPUT_DIR", output_root() / "release_clocks")
SOURCE = SCAN_INPUT.resolve()
INPUT = CLOCK_INPUT.resolve()
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
PILOT = ROOT / "pilot"
RAW = ROOT / "raw"
INPUT_RAW = INPUT / "raw"
CLOCK_DATA = ROOT / "parameter_clocks" / "data"
CLOCK_INPUT_DATA = INPUT / "parameter_clocks" / "data"
CLOCK_FIGURES = ROOT / "parameter_clocks" / "figures"


def ensure_output_dirs():
    """Create generated clock data/figure/cache subdirectories; configured input directories remain read-only."""
    for directory in (DATA, FIGURES, PILOT, RAW, CLOCK_DATA, CLOCK_FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
