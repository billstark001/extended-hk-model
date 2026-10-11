"""Lazy paths; importing this module creates no directories."""

from pathlib import Path

from common.settings import register_path

OUTPUT = register_path("EHK_PAPER_THEORY_OUTPUT_DIR", "outputs/experiments/paper_theory")
MESO = register_path("EHK_MESO_SOURCE_DIR", "../social-media-mesoscopic-models")
MICRO = register_path("EHK_MICRO_SOURCE_DIR", "../social-media-models")

SCAN_INPUT = register_path("EHK_RELEASE_SCAN_INPUT_DIR", "outputs/inputs/paper_theory/energy_scan/expanded")
CLOCK_INPUT = register_path("EHK_RELEASE_CLOCK_INPUT_DIR", "outputs/inputs/paper_theory/release_clocks")
REVISION_INPUT = register_path("EHK_REVISION_INPUT_DIR", "outputs/inputs/paper_theory/revision")
REVIEW_INPUT = register_path("EHK_REVISION_REVIEW_INPUT_DIR", "outputs/inputs/paper_theory/revision/review")
MANUSCRIPT_INPUT = register_path("EHK_MANUSCRIPT_INPUT_DIR", "outputs/inputs/paper_theory/manuscript")
KINETIC = register_path("EHK_KINETIC_BINARY", "../social-media-mesoscopic-models/bin/smp-kinetic")


def output_root(value=None):
    """Resolve an explicit generated-output path, or the configured model output root."""
    return Path(value).expanduser().resolve() if value else OUTPUT.resolve()


def input_root(value, configured):
    """Resolve an explicit input path, otherwise evaluate the supplied lazy PathSetting."""
    return Path(value).expanduser().resolve() if value else configured.resolve()
