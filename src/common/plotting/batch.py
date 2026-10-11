"""Explicit batch plotting setup and figure saving."""


def plotting():
    """Select the Agg backend explicitly and return pyplot for batch rendering."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def savefig(fig, out, name):
    """Create out, save constrained-layout PDF/PNG copies of fig, then close it."""
    out.mkdir(parents=True, exist_ok=True)
    fig.set_layout_engine("constrained")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=180)
    plotting().close(fig)
