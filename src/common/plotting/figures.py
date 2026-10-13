"""Save figures without changing decimal-containing stems."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


def save(fig, path):
    """Save PDF/PNG and close fig; path is a stem, including any decimal points, not a filename extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(path.parent / (path.name + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


def decorate(axes):
    """Label panels in reading order, hide upper spines and add a faint grid."""
    for i, ax in enumerate(np.asarray(axes).flat):
        ax.text(0.01, 1.04, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)


def numeric_log_ticks(ax):
    """Label a sparse logarithmic y-axis with ordinary numeric values."""
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
