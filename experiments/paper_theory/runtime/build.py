"""Select the study adapter sources for the shared build service."""

from pathlib import Path

from common.runtime.go import build_adapter


def build(destination, meso_root, micro_root, backend, *, link_stage=None):
    """Bind the two runtime checkouts to this study's JSONL adapter."""
    return build_adapter(
        destination, meso_root, micro_root, backend, Path(__file__).resolve().parent / "engine", link_stage=link_stage
    )
