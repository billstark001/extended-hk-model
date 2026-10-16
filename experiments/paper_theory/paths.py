"""Lazy paths; importing this module creates no directories."""

from pathlib import Path

from common.settings import register_path

OUTPUT = register_path("EHK_PAPER_THEORY_OUTPUT_DIR", "outputs/experiments/paper_theory")


def output_root(value=None):
    """Resolve an explicit generated-output path, or the configured model output root."""
    return Path(value).expanduser().resolve() if value else OUTPUT.resolve()
