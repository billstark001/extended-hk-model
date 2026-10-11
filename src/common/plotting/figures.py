"""Save figures without changing decimal-containing stems."""

import matplotlib.pyplot as plt


def save(fig, path):
    """Save PDF/PNG and close fig; path is a stem, including any decimal points, not a filename extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(path.parent / (path.name + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)
