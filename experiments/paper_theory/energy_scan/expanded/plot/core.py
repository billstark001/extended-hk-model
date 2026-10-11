"""Shared generated PDF/PNG export for expanded-scan figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

from experiments.paper_theory.energy_scan.expanded import config


def save_figure(figure: plt.Figure, name: str) -> None:
    """Create config.FIGURES, save a tightly cropped PDF/PNG pair at 240 dpi and close figure."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(config.FIGURES / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(figure)
